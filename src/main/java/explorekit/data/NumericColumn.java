package explorekit.data;

import java.io.Serializable;

/**
 * Created by giladkatz on 11/02/2016.
 */
public class NumericColumn implements Column,Serializable {
    private double[] values;

    public NumericColumn(int size) {
        values = new double[size];
    }

    public Object getValue(int i) {
        return values[i];
    }

    public void setValue(int i, Object obj) {
        values[i] = (Double)obj;
    }

    public columnType getType() {return columnType.Numeric;}

    public int getNumOfInstances() {
        return values.length;
    }

    public void setValue(int i, double v) {
        values[i] = v;
    }

    public Object getValues() {return values;}
}
