package explorekit.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.TreeMap;

/**
 * Created by giladkatz on 02/04/2016.
 */
public class DateColumn implements Column,Serializable {
    private Date[] values;
    String dateFomat;

    /**
     * This map is required as this type of columns is used for sliding window attribute generation.
     * Since this is the case, we need to be able to sort by it.
     */
    private TreeMap<Date, List<Integer>> indicesByDate = new TreeMap<>();

    public DateColumn(int size, String dateFomat) {
        values = new Date[size];
        this.dateFomat = dateFomat;
    }

    public String getDateFomat() {
        return dateFomat;
    }

    @Override
    public Object getValue(int i) {
        return values[i];
    }

    @Override
    public void setValue(int i, Object obj) {
        Date val = (Date)obj;
        values[i] = val;
        if (!indicesByDate.containsKey(val)) {
            indicesByDate.put(val, new ArrayList<>());
        }
        indicesByDate.get(val).add(i);
    }

    @Override
    public int getNumOfInstances() {
        return values.length;
    }

    @Override
    public columnType getType() {
        return columnType.Date;
    }

    public TreeMap<Date, List<Integer>> getIndicesByDate() {
        return this.indicesByDate;
    }

    public Object getValues() {return values;}
}
