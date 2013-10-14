package ml.projectthree;

import ml.ARFFParser;
import ml.Filter;
import ml.Imputer;
import ml.Matrix;

import java.io.IOException;




public class Main {

    public static final int k = 1;

    public static void main(String[] args) throws IOException {
        final int featuresStart = 0, featuresEnd = 22;
        final int labelsStart = 22, labelsEnd = 23;

        Matrix points = ARFFParser.loadARFF(args[0]);
        Matrix features = points.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = points.subMatrixCols(labelsStart, labelsEnd);

        RandomDecisionTreeLearner randomDecisionTreeLearner = new RandomDecisionTreeLearner(k);
        //EntropyReducingDecisionTreeLearner entropyReducingDecisionTreeLearner = new EntropyReducingDecisionTreeLearner(k);

        Imputer imputer = new Imputer();
        //Filter filter = new Filter(entropyReducingDecisionTreeLearner, imputer, true);
        Filter filter = new Filter(randomDecisionTreeLearner, imputer, true);

        filter.train(features, labels);

        System.out.print(randomDecisionTreeLearner.getTreeString());
        //System.out.print(entropyReducingDecisionTreeLearner.getTreeString());
    }
}
