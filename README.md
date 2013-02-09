DenoisingAutoEncoder
====================

Denoising Autoencoder with dropouts and gaussian noise for learning high level representation of the feature space in an unsupervised fashion.


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
    <img src="https://raw.github.com/ramarlina/DenoisingAutoEncoder/master/somDA_SP_500k.png" alt="results" />
    
    <p>Feature detectors after 1 million iterations with Salt and Pepper Noise: </p>
    <img src="https://raw.github.com/ramarlina/DenoisingAutoEncoder/master/somDA_1000k_MSE_0.2_SP.png" alt="results" />

</div>
