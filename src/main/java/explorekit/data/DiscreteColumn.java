package explorekit.data;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.HashMap;

/**
 * Created by giladkatz on 11/02/2016.
 */
public class DiscreteColumn implements Column,Serializable {
    private int[] values;
    private int numOfPossibleValues;

    public DiscreteColumn(int size, int numOfPossibleValues) {
        values = new int[size];
        this.numOfPossibleValues = numOfPossibleValues;
    }

    public Object getValue(int i) { return values[i]; }

    public void setValue(int i, Object obj) {
        values[i] = (Integer)obj;
    }

    public columnType getType() {return columnType.Discrete;}

    public int getNumOfInstances() {
        return values.length;
    }

    public int getNumOfPossibleValues() {
        return this.numOfPossibleValues;
    }

    public Object getValues() {return values;}

}
