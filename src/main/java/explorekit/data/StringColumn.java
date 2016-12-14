package explorekit.data;

import java.io.Serializable;

/**
 * Created by giladkatz on 07/04/2016.
 */
public class StringColumn implements Column,Serializable {
    private String[] values;
    private int numOfPossibleValues;

    public StringColumn (int size) {
        values = new String[size];
    }

    public Object getValue(int i) { return values[i]; }

    public void setValue(int i, Object obj) {
        values[i] = (String)obj;
    }

    public Column.columnType getType() {return Column.columnType.String;}

    public int getNumOfInstances() {
        return values.length;
    }

    public int getNumOfPossibleValues() {
        return this.numOfPossibleValues;
    }

    public Object getValues() {return values;}

}