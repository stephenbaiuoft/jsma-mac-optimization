"""
1. jsma = SaliencyMapMethod(model, back ='tf', sess= ses)
2. jsma.generate(preds, **jsma_params)

    # return [] of 10 classes, each element is
    # jocabian vector of 784: [g_Fc1/dx1,g_Fc1/dx2, ......g_Fc1/dx784 ]
    1. grads = jacobian_graph(preds, x, nb_classses) # define tensor relation

    # applies JSMA to batch of inputs: x is placeholder,
    # x_val is x => filled X (n x 784), but adversarila_x
    2. jsma_batch(self.sess, x, preds, grads, x_val,
                                  self.theta, self.gamma, self.clip_min,
                                  self.clip_max, nb_classes,
                                  y_target=None)
         3. ==> In the for-loop: each value in X!!!!! is jsma once!!!!!
         X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target),
                                theta, gamma, clip_min, clip_max, feed=feed)


!!!Crafting Adversarial Samples!!!!!!!!!! As the paper has suggested Algorithm
# for each GIVEN Sample!!!!!
3. jsma(sess, x, predictions, grads, sample, target, theta, gamma, clip_min,
         clip_max, feed=None):

     3.a While Loop for Maintaining JSMA Condition!!!!
         while (current != target and iteration < max_iters and
                len(search_domain) > 1):

         # call tf.run to get jacobian data iteratively... need
         # to apply changes
         3.a.1.  grads_target, grads_others
                     =  jacobian(sess, x, grads, target,adv_x_original_shape,
                        nb_features, nb_classes, feed=feed)






"""


# build Jacobian graph for tf session
def jacobian_graph(predictions, x, nb_classes):
    """"
        Create the Jacobian graph to be ran later in a TF session
        :param predictions: the model's symbolic output (linear output,
            pre-softmax)
        :param x: the input placeholder
        :param nb_classes: the number of classes the model has
        :return:
    """
    # return a list of TF gradients

    pass

# the grads from jacobian_graph ==> eventually feed into jacobian
def jacobian(sess, x, grads, target, X, nb_features, nb_classes, feed=None):
    # TensorFlow implementation of the foward derivative / Jacobian

    # Prepare feeding dictionary for all gradient computations
    feed_dict = {x: X}
    # Initialize a numpy array to hold the Jacobian component values
    # 10 x 784 (28 x 28)
    jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)
    # compute the gradients for all nb_classes
    for class_ind, grad in enumerate(grads):
        # run the each class gradient descent  F_yi/x, for each data points?
        run_grad = sess.run(grad, feed_dict)

        #
        jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))


    pass
