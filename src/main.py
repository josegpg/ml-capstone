import networks

networks.print_dataset_statistics()

########## NETWORKS TESTS

########### RANDOM
print("Training Random Model")
random_model = networks.train_random_model()
test_accuracy_random = 100 * networks.get_random_model_accuracy(random_model)
print('RandomModel Test accuracy: %.4f%%' % test_accuracy_random)

networks.test_random_model(random_model, '../faces_aligned/valid/33/image_A1861.png', 33)
print("")


########### BASIC NET
print("Training BasicNet model")
basic_model = networks.train_basic_net()
test_accuracy_basic = 100 * networks.get_basic_net_accuracy(basic_model)
print('BasicNet Test accuracy: %.4f%%' % test_accuracy_basic)

networks.test_basic_net(basic_model, '../faces_aligned/valid/33/image_A1861.png', 33)
print("")


########### TEST VGGFACE
print("Training Last Layers from VGGFace")
vggface_model = networks.train_vggface_net()
test_accuracy_vggface = 100 * networks.get_vggface_net_accuracy(vggface_model)
print('VGGFace TransferLearning Test accuracy: %.4f%%' % test_accuracy_vggface)

networks.test_vggface_net(vggface_model, '../faces_aligned/valid/33/image_A1861.png', 33)
print("")