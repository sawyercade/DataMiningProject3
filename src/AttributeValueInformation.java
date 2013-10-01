import ml.ColumnAttributes;

public class AttributeValueInformation implements Comparable<AttributeValueInformation>{
    private ColumnAttributes.ColumnType columnType;
    private Integer columnIndex;
    private Double value;
    private Double information;

    public AttributeValueInformation(ColumnAttributes.ColumnType columnType, Integer columnIndex, Double value, Double information){
        this.columnType = columnType;
        this.columnIndex = columnIndex;
        this.value = value;
        this.information = information;
    }

    //GETTERS AND SETTERS
    public ColumnAttributes.ColumnType getColumnType() {
        return columnType;
    }

    public void setColumnType(ColumnAttributes.ColumnType columnType) {
        this.columnType = columnType;
    }

    public Integer getColumnIndex() {
        return columnIndex;
    }

    public void setColumnIndex(Integer columnIndex) {
        this.columnIndex = columnIndex;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Double getInformation() {
        return information;
    }

    public void setInformation(Double information) {
        this.information = information;
    }

    @Override
    public int compareTo(AttributeValueInformation o) {
        return Double.compare(this.information, o.getInformation());
    }
}
