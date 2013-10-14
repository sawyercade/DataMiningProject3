package ml.projectthree;

import com.google.common.collect.MinMaxPriorityQueue;
import helpers.Counter;
import ml.ColumnAttributes;
import ml.MLException;
import ml.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class DecisionTreeNode {
    private static int nodeCounter;
    private Matrix features; //the features matrix for the ml.projectthree.DecisionTree this node belongs to
    private Matrix labels; //the labels matrix for the ml.projectthree.DecisionTree this node belongs to

    private List<Integer> rows; //contains column indices to the matrices that this.features and this.labels reference
    private DecisionTreeNode leftChild;
    private DecisionTreeNode rightChild;

    private SplitInformation splitInfo;
    private ColumnAttributes labelAttributes;
    private Double labelValue;

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

    public boolean isLeaf(){
        return this.leftChild==null && this.rightChild==null;
    }


    //TODO: add a way to keep from splitting on attribute values that have already been split on
    public void splitOnEntropy(int k){

        if (rows.size()>k && isHeterogenous()){ //if n>k and labels are heterogenous, split.
            MinMaxPriorityQueue<SplitInformation> infoPriorityQueue = MinMaxPriorityQueue.create(); //priority queue to store attribute-value-entropy data

            //for each column in features
            for (int colIndex = 0; colIndex < features.getNumCols(); colIndex++){
                //If this column is categorical, split on every possible attribute value for this column and compare entropies
                if(features.getColumnType(colIndex)== ColumnAttributes.ColumnType.CATEGORICAL){
                    ColumnAttributes columnAttributes = features.getColumnAttributes(colIndex); //get the possible attribute values for this column
                    for (int attributeValue = 0; attributeValue < columnAttributes.size(); attributeValue++){ //loop over attribute values
                        List<List<Integer>> splitSectionIndices = splitCategorical(colIndex, attributeValue); //split on this attribute value
                        Double information = CalculateInformation(splitSectionIndices); //calculate the information resulting from splitting on this attribute value

                        infoPriorityQueue.add(new SplitInformation(ColumnAttributes.ColumnType.CATEGORICAL, colIndex, (double)attributeValue, information)); //store the column-value-info in a min-heap keyed on info
                    }
                }
                else{
                    List<Double> columnSplitValues = getRandomSample(colIndex);
                    for (Double splitValue : columnSplitValues){ //loop over split values
                        List<List<Integer>> splitSectionIndices = splitContinuous(colIndex, splitValue); //split on splitValue
                        Double information = CalculateInformation(splitSectionIndices); //calculate the information resulting from splitting on splitValue

                        infoPriorityQueue.add(new SplitInformation(ColumnAttributes.ColumnType.CONTINUOUS, colIndex, splitValue, information));
                    }
                }
            }
            this.splitInfo = infoPriorityQueue.peek(); //get the element with the smallest entropy in the priority queue
            if(this.splitInfo.getInformation() < CalculateEntropy(this.rows, this.labels.getColumnAttributes(0))){ //if splitting is actually going to improve information, recurse
                List<List<Integer>> splitSectionIndices = this.splitInfo.getColumnType() == ColumnAttributes.ColumnType.CATEGORICAL ? splitCategorical(this.splitInfo.getColumnIndex(), (this.splitInfo.getValue().intValue())) :  splitContinuous(this.splitInfo.getColumnIndex(), this.splitInfo.getValue());
                this.leftChild = new DecisionTreeNode(features, labels, splitSectionIndices.get(0));
                this.rightChild = new DecisionTreeNode(features, labels, splitSectionIndices.get(1));

                this.leftChild.splitOnEntropy(k);
                this.rightChild.splitOnEntropy(k);
            }

            this.labelAttributes = this.labels.getColumnAttributes(0);
            this.labelValue = baselineValue(this.labels);
        }
    }

    public void splitRandom(int k){
        //if n>k and labels are heterogenous, split.
        if (rows.size()>k && isHeterogenous()){
            Random random = new Random(System.currentTimeMillis()); //seed a random number generator using current time
            boolean foundSplit = false;
            while (!foundSplit){
                int randColumn = random.nextInt(features.getNumCols()); //get a random column index
                if (features.getColumnAttributes(randColumn).getColumnType()== ColumnAttributes.ColumnType.CATEGORICAL){ //if the column is categorical
                    int randValue = random.nextInt(features.getColumnAttributes(randColumn).getValues().size()); //choose a random value from that column
                    List<List<Integer>> splitSectionIndices = splitCategorical(randColumn, randValue);
                    if (splitSectionIndices.get(0).size()>0 && splitSectionIndices.get(1).size()>0){ //if this split does not result in an empty child
                        this.splitInfo = new SplitInformation(ColumnAttributes.ColumnType.CATEGORICAL, randColumn, (double)randValue, Matrix.UNKNOWN_VALUE);
                        this.leftChild = new DecisionTreeNode(features, labels, splitSectionIndices.get(0));
                        this.rightChild = new DecisionTreeNode(features, labels, splitSectionIndices.get(1));

                        this.leftChild.splitRandom(k);
                        this.rightChild.splitRandom(k);

                        this.labelAttributes = this.labels.getColumnAttributes(0);
                        this.labelValue = baselineValue(this.labels);
                        foundSplit = true; //exit the while loop
                    }
                }
                else {
                    throw new MLException("Support for random divisions on continuous features has not been added yet");
                }
            }
        }
    }

    public String treeToString(StringBuilder output, StringBuilder prefix, String parentValue){
        if (!this.isLeaf()){
            String attributeName = this.features.getColumnAttributes(splitInfo.getColumnIndex()).getName();
            String valueName = this.features.getColumnAttributes(splitInfo.getColumnIndex()).getValue(splitInfo.getValue().intValue());

            for (int n = 0; n + 1 < prefix.length(); n++){
                output.append(prefix.charAt(n));
            }
            output.append("|\n");
            for (int n = 0; n + 1 < prefix.length(); n++){
                output.append(prefix.charAt(n));
            }
            output.append("+" + parentValue + "->Is " + attributeName + " == " + valueName + "?\n");
            prefix.append("   |");
            this.leftChild.treeToString(output, prefix, "Yes");
            prefix.deleteCharAt(prefix.length() - 1);
            prefix.append(' ');
            this.rightChild.treeToString(output, prefix, "No");
            prefix.delete(prefix.length()-4, prefix.length());
        }
        else{
            for (int n = 0; n + 1 < prefix.length(); n++){
                output.append(prefix.charAt(n));
            }
            output.append("|\n");
            for (int n = 0; n + 1 < prefix.length(); n++){
                output.append(prefix.charAt(n));
            }
            String labelVector = this.labels.getColumnAttributes(0).getValue(this.labels.getRow(this.rows.get(0)).get(0).intValue()); //get the string name of the first row's label in this.rows
            //The above only necessarily works if this node has a size of 1.
            output.append("+" + parentValue + "->'class'=" + labelVector +"\n");
        }
        return output.toString();
    }

    /**
     * Checks to see if there is more than one label in the rows indicated by this.rows.
     * @return
     */
    private boolean isHeterogenous(){
        Double value = labels.getRow(0).get(0);

        for (Integer row : rows){
            if (!labels.getRow(row).get(0).equals(value)){
                return true;
            }
        }
        return false;
    }

    /**
     * Returns a List<Double> of a sample of 8 (or rows.size() if less than 8) values taken from the rows in this node at column colIndex
     * @param colIndex
     * @return
     */
    private List<Double> getRandomSample(int colIndex) {
        int sampleSize = rows.size() < 8 ? rows.size() : 8; //if there aren't 8 rows in this node, just use each row's value in colIndex
        List<Double> splitValues = new ArrayList<Double>();

        Random random = new Random((long)rows.size()); //use the number of rows in this node as the seed
        for (int i = 0; i < sampleSize; i++){
            int rand = random.nextInt(rows.size());
            splitValues.add(features.getRow(rows.get(rand)).get(colIndex));
        }

        return splitValues;
    }

    /**
     * Returns a List of two List<Integer>, representing the rows in this.rows split on the column-attributeValue pair.
     * @param columnIndex
     * @param attributeValue
     * @return a List<List<Integer>> of size 2, where index 0 is the matching row indices, and index 1 is the non-matching row indices
     */
    private List<List<Integer>> splitCategorical(Integer columnIndex, Integer attributeValue){
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

    /**
     * Returns a List of two List<Integer>s, representing the rows in this.rows split on the column-splitValue pair
     * @param colIndex
     * @param splitValue
     * @return a List<List<Integer>> of size 2, where index 0 is the <= row indices, and index 1 is the > indices
     */
    private List<List<Integer>> splitContinuous(int colIndex, Double splitValue) {
        List<List<Integer>> sections = new ArrayList<List<Integer>>();
        List<Integer> matchingRowIndices = new ArrayList<Integer>(); //the list of rows where row.get(columnIndex) <= attributeValue
        List<Integer> nonmatchingRowIndices = new ArrayList<Integer>(); //the list of rows where row.get(columnIndex) > attributeValue

        //populate the List<List<Integer>>
        sections.add(0, matchingRowIndices);
        sections.add(1, nonmatchingRowIndices);

        //Populate the two List<Integer>s
        for (Integer rowIndex : this.rows){
            if (this.features.getRow(rowIndex).get(colIndex)<=splitValue){
                matchingRowIndices.add(rowIndex);
            }
            else{
                nonmatchingRowIndices.add(rowIndex);
            }
        }
        return sections;
    }

    private Double CalculateEntropy(List<Integer> rows, ColumnAttributes labelColumnAttributes){
        Counter<Double> counter = new Counter<Double>(); //counts the number of times each possible label attribute-value occurs
        if (rows.size()==0){
            return Double.POSITIVE_INFINITY;
        }
        Double rowSize = (double)rows.size();


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
            if(!entry.getValue().equals(0)){
                entropy += ((-1.0)*(entry.getValue()/rowSize))*(Math.log(entry.getValue()*1.0/rowSize)/Math.log(2.0));
            }
        }

        return entropy;
    }

    private Double CalculateInformation(List<List<Integer>> sections){
        Double matchingRatio = ((double)sections.get(0).size())/rows.size();
        Double nonmatchingRatio = ((double)sections.get(1).size())/rows.size();

        Double matchingEntropy = CalculateEntropy(sections.get(0), labels.getColumnAttributes(0));
        Double nonmatchingEntropy = CalculateEntropy(sections.get(1), labels.getColumnAttributes(0));

        return (matchingRatio*matchingEntropy)+(nonmatchingRatio*nonmatchingEntropy);
    }

    /**
     * Gets the most commonly occurring value in the first label column
     * @param labels
     * @return
     */
    private Double baselineValue(Matrix labels){
        Counter<Double> counts = new Counter<Double>();

        for(List<Double> row : labels.getData()){
            counts.increment(row.get(0));
        }

        return counts.getMax();
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

    public DecisionTreeNode getRightChild() {
        return rightChild;
    }

    public void setRightChild(DecisionTreeNode rightChild) {
        this.rightChild = rightChild;
    }

    public ColumnAttributes getLabelAttributes() {
        return labelAttributes;
    }

    public void setLabelAttributes(ColumnAttributes labelAttributes) {
        this.labelAttributes = labelAttributes;
    }

    public SplitInformation getSplitInfo() {
        return splitInfo;
    }

    public void setSplitInfo(SplitInformation splitInfo) {
        this.splitInfo = splitInfo;
    }

}
