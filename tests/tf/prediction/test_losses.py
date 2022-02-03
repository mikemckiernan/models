#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import tensorflow as tf

import merlin_models.tf as ml


def test_bpr():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr = ml.BPR_Loss()
    loss = bpr(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_bpr_no_reduction():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr = ml.BPR_Loss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr(targets, predictions)
    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    assert tf.reduce_mean(loss) > 0


def test_bpr_with_sample_weights():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)
    sample_weights = tf.range(1, 101, dtype=tf.float32)

    bpr_max = ml.BPR_Loss(reduction=tf.keras.losses.Reduction.NONE)
    loss = bpr_max(targets, predictions)
    loss_with_sampled_weights = bpr_max(targets, predictions, sample_weights)

    tf.assert_equal(tf.shape(loss), (batch_size, num_samples - 1))
    tf.assert_equal(tf.shape(loss_with_sampled_weights), (batch_size, num_samples - 1))

    tf.assert_equal(loss * tf.expand_dims(sample_weights, -1), loss_with_sampled_weights)

    assert (tf.reduce_mean(loss) * tf.reduce_mean(sample_weights)).numpy() == pytest.approx(
        tf.reduce_mean(loss_with_sampled_weights).numpy(), 0.05
    )


def test_bpr_multiple_positive():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 2), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 2), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr = ml.BPR_Loss()

    with pytest.raises(Exception) as excinfo:
        _ = bpr(targets, predictions)
    assert "Only one positive label is allowed per example" in str(excinfo.value)


def test_bpr_max():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    bpr_max = ml.BPRmax_Loss()
    loss = bpr_max(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_top1():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    top1 = ml.TOP1_Loss()
    loss = top1(targets, predictions)

    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_top1_v2():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    top1v2 = ml.TOP1v2_Loss()
    loss = top1v2(targets, predictions)

    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_top1_max():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    top1_max = ml.TOP1max_Loss()
    loss = top1_max(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_log_loss_max():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    log_loss = ml.Log_Loss()
    loss = log_loss(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_hinge_loss_max():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    hinge_loss = ml.Hinge_Loss()
    loss = hinge_loss(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0


def test_adaptive_hinge_loss_max():
    batch_size = 100
    num_samples = 20
    predictions = tf.random.uniform(shape=(batch_size, num_samples), dtype=tf.float32)
    positives = tf.ones(shape=(batch_size, 1), dtype=tf.float32)
    negatives = tf.zeros(shape=(batch_size, num_samples - 1), dtype=tf.float32)
    targets = tf.concat([positives, negatives], axis=1)

    adaptive_hinge_loss = ml.AdaptiveHinge_Loss()
    loss = adaptive_hinge_loss(targets, predictions)
    assert len(tf.shape(loss)) == 0
    assert loss > 0
