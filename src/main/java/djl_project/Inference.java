package djl_project;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Blocks;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.Batchifier;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Inference {
    public static void main(String[] args) throws Exception {
        
        String url = "https://resources.djl.ai/images/0.png";
        Image img = ImageFactory.getInstance().fromUrl(url);
        System.out.println("Image loaded successfully.");

        SequentialBlock block = new SequentialBlock();
        block.add(Blocks.batchFlattenBlock(28 * 28));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(10).build());

        Path modelDir = Paths.get("build/model");
        
        // --- DÜZELTİLEN KISIM BURASI ---
        // HelloDJL sınıfında "mlp-model" olarak kaydettiğin için burada da öyle çağırıyoruz.
        Model model = Model.newInstance("mlp-model"); 
        // -------------------------------
        
        model.setBlock(block);
        model.load(modelDir);
        System.out.println("Model loaded from: " + modelDir.toAbsolutePath());

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {
            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };

        try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
            Classifications result = predictor.predict(img);
            System.out.println("\n--- FINAL PREDICTION ---");
            System.out.println(result.toString());
        }
        
        model.close();
    }
}