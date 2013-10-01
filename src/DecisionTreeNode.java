import com.google.common.collect.MinMaxPriorityQueue;
import helpers.Counter;
import ml.ColumnAttributes;
import ml.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class DecisionTreeNode {

    public enum SplitType {
        CONTINUOUS,
        CATEGORICAL
    }

    private Matrix features; //the features matrix for the DecisionTree this node belongs to
    private Matrix labels; //the labels matrix for the DecisionTree this node belongs to

    private List<Integer> rows; //contains column indices to the matrices that this.features and this.labels reference
    private DecisionTreeNode leftChild;
    private DecisionTreeNode rightChild;
    private SplitType splitType; //continuous or categorical split
    private int splitColumn; //the index of the column we're splitting on
    private double continuousSplitValue; //always splits on left child less than or equal to this attribute value, right child greater than
    private String categoricalSplitValue; //left child contains this attribute value in splitColumn, right child does not

    public DecisionTreeNode(Matrix features, Matrix labels, List<Integer> rows){
        this.features = features;
        this.labels = labels;
        this.rows = rows;
    }

    public DecisionTreeNode(Matrix features, Matrix labels){
        this.features = features;
        this.labels = labels;
        this.rows = new ArrayList<Integer>();

        for (int i = 0; i < features.getNumRows(); i++){
            this.rows.add(i);
        }
    }


    public void splitOnEntropy(int k){
        //for each column in features
        for (int colIndex = 0; colIndex < features.getNumCols(); colIndex++){
            //If this column is categorical, split on every possible attribute value for this column and compare entropies
            MinMaxPriorityQueue<AttributeValueInformation> infoPriorityQueue = MinMaxPriorityQueue.create();
            if(features.getColumnType(colIndex)== ColumnAttributes.ColumnType.CATEGORICAL){
                ColumnAttributes columnAttributes = features.getColumnAttributes(colIndex); //get the possible attribute values for this column
                for (int attributeValue = 0; attributeValue < columnAttributes.size(); attributeValue++){ //loop over attribute values
                    List<List<Integer>> splitSectionIndices = splitCategoricalOnAttributeValue(colIndex, attributeValue); //split on this attribute value
                    Double information = CalculateCategoricalInformation(splitSectionIndices); //calculate the information resulting from splitting on this attribute value

                    infoPriorityQueue.add(new AttributeValueInformation(ColumnAttributes.ColumnType.CATEGORICAL, colIndex, (double)attributeValue, information)); //store the column-value-info in a min-heap keyed on info
                }
            }
            else{
                //TODO: If this column is continuous, sample 8 random values from the column and attempt to split on those. Compare entropies.
            }



        }
    }

    /**
     * Returns a List of two List<Integer>, representing the rows in this.rows split on the column-attributeValue pair.
     * @param columnIndex
     * @param attributeValue
     * @return a List<List<Integer>> of size 2, where index 0 is the matching row indices, and index 1 is the non-matching row indices
     */
    private List<List<Integer>> splitCategoricalOnAttributeValue(Integer columnIndex, Integer attributeValue){
        List<List<Integer>> sections = new ArrayList<List<Integer>>();
        List<Integer> matchingRowIndices = new ArrayList<Integer>(); //the list of rows where row.get(columnIndex) == attributeValue
        List<Integer> nonmatchingRowIndices = new ArrayList<Integer>(); //the list of rows where row.get(columnIndex) != attributeValue

        //populate the List<List<Integer>>
        sections.add(0, matchingRowIndices);
        sections.add(1, nonmatchingRowIndices);

        //Populate the two List<Integer>s
        for(Integer rowIndex : this.rows){
            if (this.features.getRow(rowIndex).get(columnIndex).intValue()==attributeValue){
                matchingRowIndices.add(rowIndex);
            }
            else {
                nonmatchingRowIndices.add(rowIndex);
            }
        }

        return sections;
    }

    private Double CalculateCategoricalEntropy(List<Integer> rows, ColumnAttributes labelColumnAttributes){
        Counter<Double> counter = new Counter<Double>();
        int numAttrValues = labelColumnAttributes.size();

        //zero the counts for each possible label attribute value
        for (int i = 0; i < labelColumnAttributes.getValues().size(); i++){
            counter.zero((double)i);
        }

        //count the number of occurrences of each attribute value in the specified rows
        for (Integer row : rows){
            counter.increment(labels.getRow(row).get(0));
        }

        Double entropy = 0.0;
        for (Map.Entry<Double, Integer> entry : counter.entries()){
            if(entry.getValue()!=0){
                entropy -= ((double)(entry.getValue()/numAttrValues))*(Math.log(entry.getValue()/numAttrValues)/Math.log(2));
            }
        }

        return entropy;
    }

    private Double CalculateCategoricalInformation(List<List<Integer>> sections){
        Double matchingRatio = ((double)sections.get(0).size())/rows.size();
        Double nonmatchingRatio = ((double)sections.get(1).size())/rows.size();

        Double matchingEntropy = CalculateCategoricalEntropy(sections.get(0), labels.getColumnAttributes(0));
        Double nonmatchingEntropy = CalculateCategoricalEntropy(sections.get(1), labels.getColumnAttributes(0));

        return (matchingRatio*matchingEntropy)+(nonmatchingRatio*nonmatchingEntropy);
    }

    //GETTERS AND SETTERS

    public Matrix getFeatures() {
        return features;
    }

    public Matrix getLabels() {
        return labels;
    }

    public List<Integer> getRows() {
        return rows;
    }

    public DecisionTreeNode getLeftChild() {
        return leftChild;
    }

    public void setLeftChild(DecisionTreeNode leftChild) {
        this.leftChild = leftChild;
    }

    public SplitType getSplitType() {
        return splitType;
    }

    public void setSplitType(SplitType splitType) {
        this.splitType = splitType;
    }

    public int getSplitColumn() {
        return splitColumn;
    }

    public void setSplitColumn(int splitColumn) {
        this.splitColumn = splitColumn;
    }

    public DecisionTreeNode getRightChild() {
        return rightChild;
    }

    public void setRightChild(DecisionTreeNode rightChild) {
        this.rightChild = rightChild;
    }

    public double getContinuousSplitValue() {
        return continuousSplitValue;
    }

    public void setContinuousSplitValue(double continuousSplitValue) {
        this.continuousSplitValue = continuousSplitValue;
    }

    public String getCategoricalSplitValue() {
        return categoricalSplitValue;
    }

    public void setCategoricalSplitValue(String categoricalSplitValue) {
        this.categoricalSplitValue = categoricalSplitValue;
    }
}
