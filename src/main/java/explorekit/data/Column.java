package explorekit.data;


/**
 * Created by giladkatz on 11/02/2016.
 */
public interface Column {

    enum columnType { Numeric, Discrete, Date, String }

    Object getValue(int i);

    void setValue(int i, Object obj);

    int getNumOfInstances();

    columnType getType();

    Object getValues();

}
