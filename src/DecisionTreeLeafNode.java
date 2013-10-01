import ml.ColumnAttributes;
import ml.Matrix;

import java.util.List;

public class DecisionTreeLeafNode extends DecisionTreeNode {
    private Matrix features;
    private Matrix labels;

//    public DecisionTreeLeafNode(List<ColumnAttributes> featuresColumnsAttributes, List<ColumnAttributes> labelsColumnAttributes) {
//        super(featuresColumnsAttributes, labelsColumnAttributes);
//    }
//
//    public DecisionTreeLeafNode(Matrix features, Matrix labels){
//        super(features.getColumnAttributes(), labels.getColumnAttributes());
//        this.features = new Matrix(features);
//        this.labels = new Matrix(labels);
//    }

    //GETTERS AND SETTERS
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
}
