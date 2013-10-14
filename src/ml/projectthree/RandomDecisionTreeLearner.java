package ml.projectthree;

import ml.ColumnAttributes;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class RandomDecisionTreeLearner extends SupervisedLearner{
    private DecisionTree decisionTree;
    private Matrix features;

    private Matrix labels;
    private Integer k;

    public RandomDecisionTreeLearner(Integer k){
        this.k = k;
    }

    @Override
    public void train(Matrix features, Matrix labels) {
        this.features = features;
        this.labels = labels;

        this.decisionTree = new DecisionTree(features, labels);
        this.decisionTree.buildRandomTree(k);
    }

    @Override
    public List<Double> predict(List<Double> in) {
        DecisionTreeNode node = decisionTree.getRoot();
        while (!node.isLeaf()){
            SplitInformation splitInfo = node.getSplitInfo();
            if (node.getSplitInfo().getColumnType()== ColumnAttributes.ColumnType.CATEGORICAL){
                if (in.get(splitInfo.getColumnIndex()).equals(splitInfo.getValue())){
                    node = node.getLeftChild();
                }
                else {
                    node = node.getRightChild();
                }
            }
            else {
                if(in.get(splitInfo.getColumnIndex())<splitInfo.getValue() || in.get(splitInfo.getColumnIndex()).equals(splitInfo.getValue())){
                    node = node.getLeftChild();
                }
                else {
                    node = node.getRightChild();
                }
            }
        }
        List<Double> prediction = new ArrayList<Double>();
        prediction.add(node.getSplitInfo().getValue());
        return prediction;
    }

    //GETTERS AND SETTERS
    public String getTreeString(){
        return decisionTree.toString();
    }

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
