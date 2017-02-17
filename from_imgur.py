"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.
"""

import numpy as np
import sys
from PIL import Image
from StringIO import StringIO
import caffe
import requests
import pika
import time
import json
import os
import datetime
from datadog import statsd

api_hostname = os.environ['API_HOSTNAME'] # URI of the API endpoint to hit
rabbitmq_hostname = os.environ['RABBITMQ_HOSTNAME']
rabbit_user = os.environ['RABBITMQ_USER']
rabbit_pass = os.environ['RABBITMQ_PASS']
s3_bucket = os.environ['S3_BUCKET']
log_debug = os.environ['LOG_LEVEL'] == 'debug'
queue_name = os.environ['QUEUE_NAME']

scores = {}

def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the
    caffe, since it was used to generate training dataset
    :param str data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    img_data = str(data)
    im = Image.open(StringIO(img_data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        resize_start = int(round(time.time() * 1000))
        img_data_rs = resize_image(pimg, sz=(256, 256))
        if (log_debug): print("RESIZE: %sms" % (int(round(time.time() * 1000)) - resize_start))

        caffe_start = int(round(time.time() * 1000))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        if (log_debug): print("CAFFE: %sms" % (int(round(time.time() * 1000)) - caffe_start))
        return outputs
    else:
        return []

def queue_msg_callback(ch, method, properties, body):
    try:
        start = int(round(time.time() * 1000))

        msg = json.loads(body)
        hash = msg['hash']
        if (log_debug): print("=== Received %s" % hash)
        if (len(hash) != 5 and len(hash) != 7):
            ch.basic_reject(delivery_tag = method.delivery_tag, requeue=False)
            statsd.increment('imgur.dev.opennsfw.reject')

        # Download image
        try:
            url = "http://%s.s3-website-us-east-1.amazonaws.com/%s" % (s3_bucket, hash)
            img_response = requests.get(url, timeout=2)
            img_response.raise_for_status()
            if (log_debug): print("DOWNLOAD: %sms" % (int(round(time.time() * 1000)) - start))
        except requests.exceptions.Timeout:
            statsd.increment('imgur.dev.opennsfw.download.timeout')
            print("Image Download Timeout for %s. Sending to dead letter queue." % hash)
            ch.basic_reject(delivery_tag = method.delivery_tag, requeue=False)
            statsd.increment('imgur.dev.opennsfw.reject')
            return
        except requests.exceptions.HTTPError:
            statsd.increment('imgur.dev.opennsfw.download.error')
            print("Image Download Status Code Error for %s. Sending to dead letter queue." % hash)
            ch.basic_reject(delivery_tag = method.delivery_tag, requeue=False)
            statsd.increment('imgur.dev.opennsfw.reject')
            return

        # Process Image
        try:
            nsfw_score = caffe_preprocess_and_compute(img_response.content, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])[1]
            scores[hash] = nsfw_score
        except:
            statsd.increment('imgur.dev.opennsfw.resize.error')
            print('Resizing error for %s. Sending to dead letter queue.' % hash)
            ch.basic_reject(delivery_tag = method.delivery_tag, requeue=False)
            statsd.increment('imgur.dev.opennsfw.reject')
            return

        ch.basic_ack(delivery_tag = method.delivery_tag)
        statsd.increment('imgur.dev.opennsfw.ack')
        if (log_debug): print("TOTAL: %sms" % (int(round(time.time() * 1000)) - start))
        if (log_debug): print("SCORE: %s" % nsfw_score)
        sys.stdout.flush()


        # Send scores back to API if cache is big enough
        if (log_debug): print("Number of cached scores: %s" % len(scores))
        if (len(scores) >= 50):
            api_success = False
            retries = 0
            while(not api_success):
                try:
                    api_start = int(round(time.time() * 1000))
                    api_response = requests.post('%s/3/moderate' % (api_hostname),
                                                 timeout=5,
                                                 data={
                                                     "scores": json.dumps(scores)
                                                 })
                    current_time = datetime.datetime.now().time()
                    if (log_debug): print("%s | API sent: %sms" % (current_time.isoformat(), int(round(time.time() * 1000)) - api_start))
                    api_response.raise_for_status()
                    global scores
                    scores={}
                    api_success = True
                except requests.exceptions.Timeout:
                    statsd.increment('imgur.dev.opennsfw.api.timeout')
                    print('API Timeout. Retrying.')
                    time.sleep(2**retries)
                    retries += 1
                except requests.exceptions.HTTPError:
                    statsd.increment('imgur.dev.opennsfw.api.error')
                    print('API Status Code Error. Retrying.')
                    time.sleep(2**retries)
                    retries += 1
    except:
        # todo: log this in sentry
        statsd.increment('imgur.dev.opennsfw.uncaught_exception')
        if (log_debug): print("Uncaught Exception!")
        ch.basic_reject(delivery_tag = method.delivery_tag, requeue=False)
        statsd.increment('imgur.dev.opennsfw.reject')

def main(argv):
    # Pre-load caffe model.
    global nsfw_net
    caffe.set_mode_gpu()
    nsfw_net = caffe.Net('nsfw_model/deploy.prototxt', 'nsfw_model/resnet_50_1by2_nsfw.caffemodel', caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    global caffe_transformer
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    ## Listen for messages from queue
    credentials = pika.PlainCredentials(rabbit_user, rabbit_pass)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_hostname,credentials=credentials))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True, arguments={"x-dead-letter-exchange" : "dead_letter_nsfw_queue"})
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue_msg_callback, queue=queue_name)
    print('[*] NSFW Worker Ready. Waiting for messages.')
    sys.stdout.flush()
    channel.start_consuming()

if __name__ == '__main__':
    main(sys.argv)