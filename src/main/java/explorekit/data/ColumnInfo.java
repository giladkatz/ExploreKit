package explorekit.data;

import explorekit.operators.Operator;

import java.io.Serializable;
import java.util.List;

/**
 * Created by giladkatz on 12/02/2016.
 */
public class ColumnInfo implements Serializable {

    public ColumnInfo(Column column, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, Class<? extends Operator> operator, String name) {
        this.column = column;
        this.sourceColumns = sourceColumns;
        this.targetColumns = targetColumns;
        this.operator = operator;
        this.name = name;
        this.isTargetClass = isTargetClass;
    }

    public Column getColumn() {
        return column;
    }

    public void setColumn(Column column) {
        this.column = column;
    }

    public void SetTargetClassValue(boolean isTargetClass){
        this.isTargetClass = isTargetClass;
    }

    public boolean isTargetClass(){
        return this.isTargetClass;
    }

    public List<ColumnInfo> getSourceColumns() {
        return this.sourceColumns;
    }

    public List<ColumnInfo> getTargetColumns() {
        return this.targetColumns;
    }

    public String getName() {
        return this.name;
    }

    private Column column;
    private List<ColumnInfo> sourceColumns;
    private List<ColumnInfo> targetColumns;
    private Class<? extends Operator> operator;
    private boolean isTargetClass = false;
    private String name;
}
