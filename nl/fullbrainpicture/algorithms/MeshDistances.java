package nl.fullbrainpicture.algorithms;


import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;



public class MeshDistances {

    // image containers
    private float[] pointList;
    private int[] triangleList;
    
    private int[] labelList;
    private float[] distance;
    private float[][] distances;
    private int[][] closest;
        
	public final void setSurfacePoints(float[] val) { pointList = val; }
	public final void setSurfaceTriangles(int[] val) { triangleList = val; }
	public final void setSurfaceLabels(int[] val) { labelList = val; }

	public final float[] 	getSurfacePoints() { return pointList; }
	public final int[] 		getSurfaceTriangles() { return triangleList; }
	public final int[] 	getLabelValues() { return labelList; }
    public final float[] 	getDistanceValues() { return distance; }
    public final float[] 	getDistanceValuesAt(int n) { return distances[n]; }
	public final int[] 	getClosestLabelsAt(int n) { return closest[n]; }
	
    public final void computeInnerDistances() {
        
        distance = MeshProcessing.computeInsideDistance(pointList, triangleList, labelList);
        
    }
	
    public final void computeFastDistances() {
        
        distance = MeshProcessing.computeInsideDistanceApproximation(pointList, triangleList, labelList);
        
    }

    public final void computeOuterDistances() {
        
        int npt = pointList.length/3;
        
        distances = new float[3][npt];
        closest = new int[3][npt];
        
        MeshProcessing.computeOutsideDistanceFunctions(3, distances, closest, pointList, triangleList, labelList);
        
    }

}
