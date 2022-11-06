import tensorflow as tf

import numpy as np
from scipy.misc import imread, imresize
from glob import glob
import matplotlib.pyplot as plt

import time 

def residual_block(x):
        
    res = tf.layers.conv2d(inputs= x, filters=128, kernel_size=3, strides=1, padding="same")    
    res = tf.layers.batch_normalization(res, axis=3, momentum=0.9, epsilon=1e-5)
    res = tf.nn.relu(res)
    res = tf.layers.conv2d(inputs= res, filters=128, kernel_size=3, strides=1, padding="same")
    res = tf.layers.batch_normalization(res, axis=3, momentum=0.9, epsilon=1e-5)
    return tf.add(res,x)
  


def build_generator(input_image, name):

    with tf.variable_scope(name):
                
        #********************************************DOWNSAMPLING*******************************************************
        #(256,256,3) 
        x = tf.layers.conv2d(inputs = input_image, filters=16, kernel_size=7, strides=1, padding="same")   
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)  
        x = tf.nn.relu(x) 
        #(256,256,16)        
        x = tf.layers.conv2d(inputs = x, filters=32, kernel_size=7, strides=2, padding="same")   
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)  
        x = tf.nn.relu(x) 
        #(128,128,32)
        x = tf.layers.conv2d(inputs= x, filters=64, kernel_size=3, strides=2, padding="same")  
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
        x = tf.nn.relu(x)
        #(64,64,64)
        x = tf.layers.conv2d(inputs= x, filters=128, kernel_size=3, strides=2, padding="same")        
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
        x = tf.nn.relu(x)
        #(32,32,128)
        #********************************************RESIDUAL*******************************************************
        for _ in range(6):
            x = residual_block(x) 
            
        #********************************************UPSAMPLING*******************************************************
        #(32,32,128)        
        x = tf.layers.conv2d_transpose(inputs= x, filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)       
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
        x = tf.nn.relu(x)
        #(64,64,64) 
        x = tf.layers.conv2d_transpose(inputs= x, filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)       
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
        x = tf.nn.relu(x)
        #(128,128,32)  
        x = tf.layers.conv2d_transpose(inputs= x, filters=16, kernel_size=7, strides=2, padding='same', use_bias=False)       
        x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
        x = tf.nn.relu(x)
        #(256,256,16)  
        x = tf.layers.conv2d(inputs= x, filters=3, kernel_size=7, strides=1, padding="same")
        output = tf.nn.tanh(x, name = "output")
                
    return output


def build_discriminator(input_image, name):

    with tf.variable_scope(name) as scope:
        
        #256 256 3
        x = tf.pad(input_image, [[0,0],[1,1],[1,1],[0,0]]) 
        #258 258 3
        x = tf.layers.conv2d(inputs= x, filters=128, kernel_size=4, strides=2, padding="valid")   
        #128 128 128
        x = tf.nn.leaky_relu(x, alpha = 0.2)
        #128 128 128
        #------------------------------------------
        x = tf.pad(input_image, [[0,0],[1,1],[1,1],[0,0]]) 
        #130 130 128
        x = tf.layers.conv2d(inputs= x, filters=64, kernel_size=4, strides=2, padding="valid")   
        #64 64 64
        x = tf.nn.leaky_relu(x, alpha = 0.2)
        #64 64 64 
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
        #66 66 64        
        for i in range(1, 4):#3 hidden layers
           
            x = tf.layers.conv2d(inputs= x, filters = 2**i*64, kernel_size=4, strides=2, padding="valid")        
            x = tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=1)
            x = tf.nn.leaky_relu(x, alpha = 0.2)
                
            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
        #10 10 512        
        output = tf.layers.conv2d(inputs= x, filters=1, kernel_size=4, strides=1, activation="sigmoid")
        #7 7 1    
    return output




def load_images(data_dir):
          
    imagesA = glob(data_dir + '/trainA/*.*')
    imagesB = glob(data_dir + '/trainB/*.*')

    allImagesA = []
    allImagesB = []

    for index, filename in enumerate(imagesA):
        imgA = imread(filename, mode='RGB')
        imgB = imread(imagesB[index], mode='RGB')

        imgA = imresize(imgA, (128, 128))
        imgB = imresize(imgB, (128, 128))

        if np.random.random() > 0.5:
            imgA = np.fliplr(imgA)
            imgB = np.fliplr(imgB)

        allImagesA.append(imgA)
        allImagesB.append(imgB)

    # Normalize images
    allImagesA = np.array(allImagesA) / 127.5 - 1. #Normalized
    allImagesB = np.array(allImagesB) / 127.5 - 1. #Normalized
    #allImagesA = np.array(allImagesA)  #Non Normalized
    #allImagesB = np.array(allImagesB)  #Non Normalized

    return allImagesA, allImagesB



def load_test_batch(data_dir, batch_size):
    
    imagesA = glob(data_dir + '/testA/*.*')
    imagesB = glob(data_dir + '/testB/*.*')

    imagesA = np.random.choice(imagesA, batch_size)
    imagesB = np.random.choice(imagesB, batch_size)

    allA = []
    allB = []

    for i in range(len(imagesA)):
        # Load images and resize images
        imgA = imresize(imread(imagesA[i], mode='RGB').astype(np.float32), (128, 128))
        imgB = imresize(imread(imagesB[i], mode='RGB').astype(np.float32), (128, 128))

        allA.append(imgA)
        allB.append(imgB)
    
    return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0 #Normalized
    #return np.array(allA) , np.array(allB) #Non Normalized



def save_images(originalA, generatedB, recosntructedA, originalB, generatedA, reconstructedB, path):

    
    #fig = plt.figure(figsize=(7.1,5.1))#18x13 7.1x5.1     5.9x3.93 = 15x10    30x20 11.8 7.86
    fig = plt.figure(figsize=(11.8,7.86))#18x13 7.1x5.1     5.9x3.93 = 15x10    30x20 11.8 7.86
    #fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(originalA)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(generatedB)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(recosntructedA)
    ax.axis("off")
    ax.set_title("Reconstructed")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(originalB)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(generatedA)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(reconstructedB)
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)

  
    

def add_summary(writer, name, value, global_step):

    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    writer.add_summary(summary, global_step=global_step)
    writer.flush()

if __name__ == '__main__':
    
    dataset_type = "facades512/"    
    dataset = "/content/drive/My Drive/VisionPerception/datasets/facades512/"

    batch_size = 1
    epochs = 50
    learningRate = 0.0002
    beta = 0.5
    lam = 10
    #mode = 'predict'
    mode = 'train'
                 
    tf.reset_default_graph()
    print("---------------------------------------------")
    print("Mode: " + mode)
    print("Dataset: " + dataset)
    print("Dataset Type : " + dataset_type)
    print("---------------------------------------------")

    if mode == 'train':

        imagesA, imagesB = load_images(dataset)
       
        input_A = tf.placeholder(tf.float32, [None, 128, 128, 3], name="input_A")
        input_B = tf.placeholder(tf.float32, [None, 128, 128, 3], name="input_B")

        with tf.variable_scope("Model") as scope:
            
            gen_B = build_generator(input_A, name="generator_AtoB")
            gen_A = build_generator(input_B, name="generator_BtoA")
            disc_A = build_discriminator(input_A, name="discriminator_A")
            disc_B = build_discriminator(input_B, name="discriminator_B")
            
            scope.reuse_variables()
            
            disc_gen_A = build_discriminator(gen_A, "discriminator_A")
            disc_gen_B = build_discriminator(gen_B, "discriminator_B")
            cyc_A = build_generator(gen_B, "generator_BtoA")
            cyc_B = build_generator(gen_A, "generator_AtoB")
                
        #https://github.com/architrathore/CycleGAN/
        #https://hardikbansal.github.io/CycleGANBlog/
        '''
        #Discriminator
        disc_A_loss = (tf.reduce_mean(tf.squared_difference(disc_A,1))+tf.reduce_mean(tf.square(disc_gen_A)))/2 #ANDREBBE DIVISA TUTTA PER 2
        disc_B_loss = (tf.reduce_mean(tf.squared_difference(disc_B,1))+tf.reduce_mean(tf.square(disc_gen_B)))/2 #ANDREBBE DIVISA TUTTA PER 2
        #Cyclic
        cyclic_loss = tf.reduce_mean(tf.abs(input_A - cyc_A)) + tf.reduce_mean(tf.abs(input_B - cyc_B))
        #Generator
        gen_A_loss = tf.reduce_mean(tf.squared_difference(disc_gen_A,1)) + lam * cyclic_loss
        gen_B_loss = tf.reduce_mean(tf.squared_difference(disc_gen_B,1)) + lam * cyclic_loss
        
        d_loss = (disc_A_loss + disc_B_loss)/2 # Total Loss discriminator
        g_loss = (gen_A_loss + gen_B_loss)/2   # Total Loss Generator
        '''
        
        
        disc_A_loss = (tf.reduce_mean(tf.squared_difference(disc_A,1)) + tf.reduce_mean(tf.square(disc_gen_A)))/2
        disc_B_loss = (tf.reduce_mean(tf.squared_difference(disc_B,1)) + tf.reduce_mean(tf.square(disc_gen_B)))/2

        cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))
        #                                                                                    identty loss
        gen_A_loss = tf.reduce_mean(tf.squared_difference(disc_gen_B,1)) + lam * cyc_loss #+ tf.reduce_mean(tf.abs(input_A - gen_B))
        gen_B_loss = tf.reduce_mean(tf.squared_difference(disc_gen_A,1)) + lam * cyc_loss #+ tf.reduce_mean(tf.abs(input_B - gen_A))
        
        
        d_loss = (disc_A_loss + disc_B_loss)/2
        g_loss = (gen_A_loss + gen_B_loss)/2
        
        '''
        ####
        #-----------------------------< LOSS WITHOUT IDENTITY LOSS FOR FACEDES1
        ####
        disc_A_loss = (tf.reduce_mean(tf.squared_difference(disc_A,1)) + tf.reduce_mean(tf.square(disc_gen_A)))/2
        disc_B_loss = (tf.reduce_mean(tf.squared_difference(disc_B,1)) + tf.reduce_mean(tf.square(disc_gen_B)))/2

        cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))
        #                                                                                    identty loss
        gen_A_loss = tf.reduce_mean(tf.squared_difference(disc_gen_B,1)) + lam * cyc_loss 
        gen_B_loss = tf.reduce_mean(tf.squared_difference(disc_gen_A,1)) + lam * cyc_loss 
        
        
        d_loss = (disc_A_loss + disc_B_loss)/2
        g_loss = (gen_A_loss + gen_B_loss)/2
        '''
        optimizer = tf.train.AdamOptimizer(learningRate, beta)
        
        model_vars = tf.trainable_variables()

        d_A_vars = [var for var in model_vars if 'discriminator_A' in var.name]
        g_A_vars = [var for var in model_vars if 'generator_A' in var.name]
        d_B_vars = [var for var in model_vars if 'discriminator_B' in var.name]
        g_B_vars = [var for var in model_vars if 'generator_B' in var.name]
        
        g_A_trainer = optimizer.minimize(gen_A_loss, var_list=g_A_vars)
        g_B_trainer = optimizer.minimize(gen_B_loss, var_list=g_B_vars)   
        d_A_trainer = optimizer.minimize(disc_A_loss,var_list=d_A_vars)
        d_B_trainer = optimizer.minimize(disc_B_loss,var_list=d_B_vars)


        saver = tf.train.Saver()
        
        with tf.Session() as sess:
          
            sess.run(tf.global_variables_initializer())
            
            writer = tf.summary.FileWriter("/content/drive/My Drive/VisionPerception/summary/" + dataset_type, sess.graph)
            
            for epoch in range(epochs):
          
                start_seconds = time.time()
                #print("Epoch:{}".format(epoch))

                dis_losses = []
                gen_losses = []
      
                num_batches = int(min(imagesA.shape[0], imagesB.shape[0]) / batch_size)
                print("Number of batches:{}".format(num_batches))
                   
                for index in range(num_batches):

                    batchA = imagesA[index * batch_size:(index + 1) * batch_size]
                    batchB = imagesB[index * batch_size:(index + 1) * batch_size]
                    _, gen_B_run = sess.run([g_A_trainer, gen_B], feed_dict={input_A: batchA, input_B: batchB})
                    _ = sess.run([d_B_trainer], feed_dict={input_A: batchA, input_B: batchB})
                    _, gen_A_run = sess.run([g_B_trainer, gen_A], feed_dict={input_A: batchA, input_B: batchB})
                    _ = sess.run([d_A_trainer], feed_dict={input_A: batchA, input_B: batchB})
                    tem_g_loss = sess.run(g_loss, feed_dict={input_A: batchA, input_B: batchB})
                    tem_d_loss = sess.run(d_loss, feed_dict={input_A: batchA, input_B: batchB})
                    dis_losses.append(tem_d_loss)
                    gen_losses.append(tem_g_loss)
                
                final_seconds = time.time()
                elapsed = final_seconds - start_seconds
                print("Time for " + str(epoch) + "° epoch = " + str(elapsed) )

                #Save model
                save_path = saver.save(sess, "/content/drive/My Drive/VisionPerception/models/"+dataset_type)
                #print("Model saved ")
                
                #Save Tensorboard
                add_summary(writer, 'Discriminator_loss', np.mean(dis_losses), epoch)
                add_summary(writer, 'Generator_loss', np.mean(gen_losses), epoch)
                add_summary(writer, 'Seconds_for_epochs', elapsed, epoch)

                if epoch % 1 == 0:

                    batchA, batchB = load_test_batch(data_dir=dataset, batch_size=2)
                    gen_B_run = sess.run( gen_B, feed_dict={input_A: batchA})
                    gen_A_run = sess.run( gen_A, feed_dict={input_B: batchB})
                    reconstructA = sess.run( gen_A, feed_dict={input_B: gen_B_run})
                    reconstructB = sess.run( gen_B, feed_dict={input_A: gen_A_run})
                    #Save results
                    for i in range(len(gen_A_run)):
                        save_images(originalA=batchA[i], generatedB = gen_B_run[i], recosntructedA=reconstructA[i],
                                    originalB=batchB[i], generatedA = gen_A_run[i], reconstructedB=reconstructB[i],
                                   path="/content/drive/My Drive/VisionPerception/image/"+ dataset_type + "gen_{}_{}".format(epoch, i))
            
    elif mode == 'predict':
    
     with tf.Session() as sess:
        images = [] 
         
        sess.run(tf.global_variables_initializer())
        new_saver= tf.train.import_meta_graph("/content/drive/My Drive/VisionPerception/models/"+dataset_type+".meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("/content/drive/My Drive/VisionPerception/models/"+dataset_type))
        graph = tf.get_default_graph()
        n_input_A = graph.get_tensor_by_name("input_A:0")
        n_input_B = graph.get_tensor_by_name("input_B:0")
        gen_B = graph.get_tensor_by_name("Model/generator_AtoB/output:0")
        gen_A = graph.get_tensor_by_name("Model/generator_BtoA/output:0")
        batchA, batchB = load_test_batch(data_dir=dataset, batch_size = 100)        
        gen_B_run = sess.run( gen_B , feed_dict={n_input_A: batchA})
        gen_A_run = sess.run( gen_A , feed_dict={n_input_B: batchB})
        reconstructA = sess.run( gen_A, feed_dict={n_input_B: gen_B_run})
        reconstructB = sess.run( gen_B, feed_dict={n_input_A: gen_A_run})
        #Save results
        for i in range(len(gen_A_run)):
            save_images(originalA=batchA[i], generatedB = gen_B_run[i], recosntructedA=reconstructA[i],
                        originalB=batchB[i], generatedA = gen_A_run[i], reconstructedB=reconstructB[i],
                        path="/content/drive/My Drive/VisionPerception/Test/"+dataset_type+"test_{}".format(i))                               

                  
