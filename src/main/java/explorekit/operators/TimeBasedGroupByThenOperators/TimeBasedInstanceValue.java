package explorekit.operators.TimeBasedGroupByThenOperators;

import java.util.Date;

/**
 * Created by giladkatz on 16/04/2016.
 */
public class TimeBasedInstanceValue {
    public Date instanceDate;
    public Object value;

    public TimeBasedInstanceValue(Date date, Object val) {
        this.instanceDate = date;
        this.value = val;
    }

    public Date getInstanceDate() {
        return this.instanceDate;
    }

    public Object getValue() {
        return this.getValue();
    }
}
