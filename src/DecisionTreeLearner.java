import ml.Matrix;
import ml.SupervisedLearner;

import java.util.List;

public class DecisionTreeLearner extends SupervisedLearner {
    private DecisionTree decisionTree;
    private Matrix features, labels;
    private Integer k;

    public DecisionTreeLearner(Integer k){
        this.k = k;
    }

    @Override
    public void train(Matrix features, Matrix labels) {
        this.features = features;
        this.labels = labels;

        decisionTree = new DecisionTree(features, labels);
        decisionTree.buildEntropyReducingTree(this.k);
    }

    @Override
    public List<Double> predict(List<Double> in) {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public String getTreeString(){
        return decisionTree.toString();
    }

    //GETTERS AND SETTERS
    public DecisionTree getDecisionTree() {
        return decisionTree;
    }

    public void setDecisionTree(DecisionTree decisionTree) {
        this.decisionTree = decisionTree;
    }

    public Matrix getFeatures() {
        return features;
    }

    public void setFeatures(Matrix features) {
        this.features = features;
    }

    public Matrix getLabels() {
        return labels;
    }

    public void setLabels(Matrix labels) {
        this.labels = labels;
    }

    public Integer getK() {
        return k;
    }

    public void setK(Integer k) {
        this.k = k;
    }
}
