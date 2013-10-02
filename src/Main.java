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

        Matrix points = ARFFParser.loadARFF("G:\\Projects\\DataMiningProject3\\mushroom.arff");
        Matrix features = points.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = points.subMatrixCols(labelsStart, labelsEnd);

        DecisionTreeLearner decisionTreeLearner = new DecisionTreeLearner(k);

        Imputer imputer = new Imputer();
        Filter filter = new Filter(decisionTreeLearner, imputer, true);

        filter.train(features, labels);

        System.out.print(decisionTreeLearner.getTreeString());
    }
}
