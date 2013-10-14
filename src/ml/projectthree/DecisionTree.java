package ml.projectthree;

import ml.MLException;
import ml.Matrix;

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
        this.root.splitOnEntropy(this.k);
        return this.root;
    }

    public DecisionTreeNode buildRandomTree(int k){
        this.k = k;
        this.root.splitRandom(this.k);
        return this.root;
    }

    @Override
    public String toString(){
        StringBuilder output = new StringBuilder();
        StringBuilder prefix = new StringBuilder();
        return this.root.treeToString(output, prefix, "");
    }

    public DecisionTreeNode getRoot() {
        return root;
    }

    public void setRoot(DecisionTreeNode root) {
        this.root = root;
    }

}
