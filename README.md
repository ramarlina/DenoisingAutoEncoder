Denoising AutoEncoder
=====================

<br/>

Denoising Autoencoder can be trained to learn high level representation of the feature space in an unsupervised fashion.
<br/>
A deep neural network can be created by stacking layers of pre-trained autoencoders one on top of the other.
The training of the whole network is done in three phases:
<div>
    1. Pre-training: In this phase, each layer is trained to reconstruct original data from corrupted version.  
        Different efficient methods of corrupting input include: <br/>
            - Adding small gaussian noises<br/>
            - Randomly set variables to arbitrary values<br/>
            - Randomly set input variables to 0<br/>
  <br/>
    2. Learning: In this phase, a sigmoid layer and a softmax layer are placed on top of the stack, and trained
       for classification tasks.
       <br/>
    3. Fine-tuning: The whole network is fine-tuned using standard backprobagation algorithm   
    <br/>
</div>
    # Create the structure of stacked denoising autoencoders
    sDA = StackedDA([300, 100])
    
    # Pre-train layers one at a time, with 50% Salt and Pepper noise
    sDA.pre_train(X[:1000], rate=0.5, n_iters=500)
    
    # Add Softmax top layer for classification
    sDA.finalLayer(y[:1000], learner_size=200, n_iters=1)
    
    # Run Backpropagation to fine-tune the whole network
    sDA.fine_tune(X[:1000], y[:1000], n_iters=1)
    
    # Predicting probabilities P(yk=1|X)
    pred = sDA.predict(X)



<div>
    <h1>Results</h1>
    <p>Feature detectors after 500k iterations with Gaussian Noise: </p>
    <img src="https://raw.github.com/ramarlina/DenoisingAutoEncoder/master/results/somDA_SP_500k.png" alt="results" />
    
    <p>Feature detectors after 1 million iterations with Salt and Pepper Noise: </p>
    <img src="https://raw.github.com/ramarlina/DenoisingAutoEncoder/master/results/somDA_1000k_MSE_0.2_SP.png" alt="results" />

</div>

Here is great lecture from Paul Vincent on denoising auto encoders: http://videolectures.net/icml08_vincent_ecrf/
<br/>
http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
