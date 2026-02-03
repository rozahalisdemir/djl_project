package djl_project;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Blocks;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;


public class HelloDJL {
    public static void main(String[] args) {
        
        // --- CONFIGURATION ---
        long inputSize = 28 * 28; // 784 flattened pixels
        long outputSize = 10;    // Digits 0 to 9
        int batchSize = 32;      // Group size for training
        int epochs = 2;          // Number of times to pass through the dataset

        try {
            // STEP 1: PREPARE DATASET
            System.out.println("Step 1: Preparing MNIST dataset...");
            Mnist mnist = Mnist.builder()
                    .setSampling(batchSize, true) 
                    .build();
            mnist.prepare(); 

            // STEP 2: BUILD NEURAL NETWORK ARCHITECTURE
            System.out.println("Step 2: Constructing the MLP architecture...");
            SequentialBlock block = new SequentialBlock();
            
            // Flatten 2D image (28x28) into 1D vector (784)
            block.add(Blocks.batchFlattenBlock(inputSize)); 
            // Hidden Layer 1: 128 units
            block.add(Linear.builder().setUnits(128).build()); 
            block.add(Activation::relu);                       
            // Hidden Layer 2: 64 units
            block.add(Linear.builder().setUnits(64).build());  
            block.add(Activation::relu);                       
            // Output Layer: 10 units
            block.add(Linear.builder().setUnits(outputSize).build()); 

            // STEP 3: INITIALIZE MODEL
            Model model = Model.newInstance("mnist-mlp-model");
            model.setBlock(block);

            // STEP 4: SETUP TRAINING CONFIGURATION
            // SoftmaxCrossEntropy is standard for classification
            TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy()) 
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            Trainer trainer = model.newTrainer(config);

            // STEP 5: INITIALIZE PARAMETERS
            // Input shape represents (Batch Size, Channels, Height, Width)
            // For MLP, we use (1, 784)
            trainer.initialize(new Shape(1, inputSize));

            // STEP 6: START TRAINING
            System.out.println("Step 6: Starting training process...");
            EasyTrain.fit(trainer, epochs, mnist, null);

            // STEP 7: SAVE TRAINED MODEL
            Path modelDir = Paths.get("build/model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
            }
            model.save(modelDir, "mlp-model");
            
            System.out.println("\nSUCCESS: Model trained and saved to 'build/model' directory.");

        } catch (Exception e) {
            System.err.println("CRITICAL ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }
}