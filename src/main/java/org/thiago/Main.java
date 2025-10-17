package org.thiago;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        long seed = 12345;
        Nd4j.getRandom().setSeed(seed);

        int numLinesToSkip = 0;
        char delimiter = ',';
        int labelIndex = 4;       // coluna da classe original
        int numClasses = 3;       // Iris tem 3 classes
        int batchSize = 150;      // tamanho total do dataset

        Schema inputSchema = new Schema.Builder()
                .addColumnsDouble("sepallength", "sepalwidth", "petallength", "petalwidth")
                .addColumnString("label")
                .build();

        List<String> categories = Arrays.asList("Iris-setosa", "Iris-versicolor", "Iris-virginica");
        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .transform(new StringToCategoricalTransform("label", categories))
                .transform(new CategoricalToIntegerTransform("label"))
                .build();

        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(new FileSplit(new File("src/main/resources/iris.data")));
        TransformProcessRecordReader tprr = new TransformProcessRecordReader(rr, tp);

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(tprr, batchSize, labelIndex, numClasses);

        org.nd4j.linalg.dataset.DataSet allData = iterator.next();
        allData.shuffle(seed);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        org.nd4j.linalg.dataset.SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        var trainingData = testAndTrain.getTrain();
        var testData = testAndTrain.getTest();

        int numInputs = 4;
        int numHidden = 50;
        int outputNum = 3;
        int maxEpochs = 10000;   // max epochs (parada automática se atingir 1.0)
        double lr = 0.005;       // taxa de aprendizado menor

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHidden).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHidden).nOut(outputNum).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Treinamento com checagem periódica da acurácia de teste e parada cedo
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // embaralhar treinamento a cada época para melhor convergência
            trainingData.shuffle(seed + epoch);
            model.fit(trainingData);

            if ((epoch + 1) % 10 == 0) {
                INDArray output = model.output(testData.getFeatures());
                Evaluation eval = new Evaluation(outputNum);
                eval.eval(testData.getLabels(), output);
                double acc = eval.accuracy();
                System.out.printf("Época %d - acurácia teste: %.4f%n", epoch + 1, acc);
                if (acc == 1.0) {
                    System.out.printf("Acurácia 1.0 alcançada na época %d%n", epoch + 1);
                    break;
                }
            }
        }

        // Avaliação final
        INDArray output = model.output(testData.getFeatures());
        Evaluation eval = new Evaluation(outputNum);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
    }
}
