# Shuttle data
The shuttle dataset contains 9 attributes all of which are numerical. The first one being time, so the input data for models contains only the rest 8 measure variables.  The last column is the class which has been coded as follows :
    	1   	Rad Flow
    	2   	Fpv Close
    	3   	Fpv Open
    	4   	High
    	5   	Bypass
    	6   	Bpv Close
    	7	 Bpv Open
Class 4 is dropped as most studies on anomaly detection studies do the same. Class 1 is normal, and the rest classes are anomalous.

The structure of input data are multiple parallel series.

Source from UCI Machine Learning Repository
