import ml.Matrix;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class DecisionTree {
    private DecisionTreeNode root;
    private Matrix features;
    private Matrix labels;
    private int k;

    public DecisionTree(final Matrix features, final Matrix labels){
        this.features = features;
        this.labels = labels;
        this.root = new DecisionTreeNode(features, labels);
    }

    public DecisionTreeNode buildEntropyReducingTree(int k){
        this.k = k;
        throw new NotImplementedException();
    }

    public DecisionTreeNode buildRandomTree(int k){
        this.k = k;
        throw new NotImplementedException();
    }

    private static double calculateEntropy(Matrix labels){
        throw new NotImplementedException();
    }
}
