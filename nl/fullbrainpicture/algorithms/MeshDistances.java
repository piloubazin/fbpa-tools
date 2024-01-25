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
    private float[] valueList;
    private float[] distance;
    private float[][] distances;
    private int[][] closest;
    
    private int depth=4;
    private boolean normalize=false;
        
	public final void setSurfacePoints(float[] val) { pointList = val; }
	public final void setSurfaceTriangles(int[] val) { triangleList = val; }
	public final void setSurfaceLabels(int[] val) { labelList = val; }
	public final void setSurfaceValues(float[] val) { valueList = val; }
	
	public final void setDistanceDepth(int val) { depth=val; }
	public final void setNormalizeWeights() { normalize=true; }

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

    public final void computeWeightedDistances() {
        
        if (normalize) {
            float max = 0.0f;
            for (int n=0;n<valueList.length;n++)
                if (valueList[n]>max) max = valueList[n];
            if (max>0) 
                for (int n=0;n<valueList.length;n++)
                    valueList[n] /= max;
        }
        
        distance = MeshProcessing.computeWeightedDistanceFunction(pointList, triangleList, valueList);
        
    }

    public final void computeDistanceWeighting(float ratio) {
        
        distance = MeshProcessing.computeInsideDistance(pointList, triangleList, labelList);
        
        int npt = pointList.length/3;
        
        // derive sparse connectivity matrix(closest neighbors only)
        int[] labels = ObjectLabeling.listLabels(labelList, npt,1,1);
        int nlb = labels.length;
       
        int[] centroid = new int[nlb];
        float[] indist = new float[nlb];
        for (int p=0;p<npt;p++) {
            if (distance[p]>0) {
                int lb=-1;
                for (int l=0;l<nlb;l++) if (labels[l]==labelList[p]) lb=l;
                if (distance[p]>indist[lb]) {
                    indist[lb] = distance[p];
                    centroid[lb] = p;
                }
            }
        }
        for (int p=0;p<npt;p++) {
            if (labelList[p]>0) {
                int lb=-1;
                for (int l=0;l<nlb;l++) if (labels[l]==labelList[p]) lb=l;
                
                distance[p] = Numerics.min(1.0f, distance[p]/(ratio*indist[lb]));
            }
        }
        
    }
	
    public final void computeOuterDistances() {
        
        int npt = pointList.length/3;
        
        distances = new float[depth][npt];
        closest = new int[depth][npt];
        
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, labelList);
        
    }

    public final void computeSignedDistances() {
        
        int npt = pointList.length/3;
        
        distances = new float[depth][npt];
        closest = new int[depth][npt];
        
        MeshProcessing.computeSignedDistanceFunctions(depth, distances, closest, pointList, triangleList, labelList);
        
    }

    public final void computeMinimumDistances() {
        
        int npt = pointList.length/3;
        
        // derive sparse connectivity matrix(closest neighbors only)
        int[] labels = ObjectLabeling.listLabels(labelList, npt,1,1);
        int nlb = labels.length;
       
        // first find region centroids
        float[] init = MeshProcessing.computeInsideDistance(pointList, triangleList, labelList);
        int[] centroid = new int[nlb];
        float[] indist = new float[nlb];
        for (int p=0;p<npt;p++) {
            if (init[p]>0) {
                int lb=-1;
                for (int l=0;l<nlb;l++) if (labels[l]==labelList[p]) lb=l;
                if (init[p]>indist[lb]) {
                    indist[lb] = init[p];
                    centroid[lb] = p;
                }
            }
        }
        
        // build a centroid-only version of the region
        int[] centroidList = new int[npt];
        for (int p=0;p<npt;p++) if (labelList[p]<0) centroidList[p] = labelList[p];
        for (int l=0;l<nlb;l++) if (labels[l]>0) centroidList[centroid[l]] = labels[l];
        
        distances = new float[depth][npt];
        closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeSignedDistanceFunctions(depth, distances, closest, pointList, triangleList, centroidList);
        
        int[][] ngb = MeshProcessing.generatePointNeighborTable(npt, triangleList);

        float[][] distmatrix = new float[nlb][nlb];
        float[] path = new float[npt];
        int[] connect = new int[npt];
        
        for (int p=0;p<npt;p++) for (int n=0;n<nlb;n++) {
            connect[p]=labelList[p];
        }
        
        for (int n1=0;n1<nlb;n1++) for (int n2=n1+1;n2<nlb;n2++) if (labels[n1]>0 && labels[n2]>0) {
            // find the shortest path between two regions
            float[] dist1 = new float[npt];
            float[] dist2 = new float[npt];
            boolean[] mask1 = new boolean[npt];
            boolean[] mask2 = new boolean[npt];
            int ndist=0;
            
            // find locations with both distances defined
            for (int p=0;p<npt;p++) {
                for (int d=0;d<depth;d++) {
                    if (closest[d][p]==labels[n1]) dist1[p] = distances[d][p];
                    else if (closest[d][p]==labels[n2]) dist2[p] = distances[d][p];
                }
                if (dist1[p]>0) mask1[p] = true;
                if (dist2[p]>0) mask2[p] = true;
                if (mask1[p] && mask2[p]) ndist++;
            }
            
            // find the closest path
            if (ndist>0) {
                // start at the closest point to both labels
                float d0 = 0.0f;
                int p0 = -1;
                for (int p=0;p<npt;p++) if (mask1[p] && mask2[p]) {
                    if (p0==-1 || (p0>-1 && (1+Numerics.abs(dist1[p]-dist2[p]))*(dist1[p]+dist2[p])<d0) ) {
                        d0 = (1+Numerics.abs(dist1[p]-dist2[p]))*(dist1[p]+dist2[p]);
                        p0 = p;
                    }
                }
                float maxdist = 0.0f;
                float dist=d0;
                int p=p0;
                // go toward first region
                while (p>-1) {
                    // set current point
                    path[p] = dist;
                    connect[p] = labels[n1];
                    if (dist>maxdist) maxdist = dist;
                    mask1[p] = false;
                    
                    int pN=-1;
                    float dN = 1e9f;
                    for (int k=0;k<ngb[p].length;k++) if (mask1[ngb[p][k]]) {
                        if (pN==-1 || (pN>-1 && dist1[ngb[p][k]]<dN) ) { 
                           pN = ngb[p][k];
                           dN = dist1[pN];
                        }
                    }
                    // stop when we are not getting closer to n2? or wait until the proper point?
                    if (pN>-1 && dist1[pN]>dist1[p]) {
                        p=-1;
                    } else {
                        p = pN;
                        dist = dist1[pN]+dist2[pN];
                    }
                }
                dist=d0;
                p=p0;
                // go toward second region
                while (p>-1) {
                    // set current point
                    path[p] = dist;
                    connect[p] = labels[n2];
                    if (dist>maxdist) maxdist = dist;
                    mask2[p] = false;
                    
                    int pN=-1;
                    float dN = 1e9f;
                    for (int k=0;k<ngb[p].length;k++) if (mask2[ngb[p][k]]) {
                        if (pN==-1 || (pN>-1 && dist2[ngb[p][k]]<dN) ) { 
                           pN = ngb[p][k];
                           dN = dist2[pN];
                        }
                    }
                    // stop when we are not getting closer to n2
                    if (pN>-1 && dist2[pN]>dist2[p]) {
                        p=-1;
                    } else {
                        p = pN;
                        dist = dist1[pN]+dist2[pN];
                    }
                }
                distmatrix[n1][n2] = maxdist;
                distmatrix[n2][n1] = maxdist;    
                
                System.out.println(" label "+labels[n1]+" - label "+labels[n2]+" = "+maxdist);
            }
        }
        // reuse the maps
        distances = new float[1][];
        closest = new int[1][];
        distances[0] = path;
        closest[0] = connect;
    }

    public final void computeValueSkeleton() {
        
        int npt = pointList.length/3;
        int[][] ngb = MeshProcessing.generatePointNeighborTable(npt, triangleList);

        // label the points with only one value above
        int[] skeleton = new int[npt];
        for (int p=0;p<npt;p++) if (valueList[p]>0) {
             int nsup=0;
             for (int n=0;n<ngb[p].length;n++) {
                 if (valueList[ngb[p][n]]>=valueList[p]) nsup++;
             }
             if (nsup==1) skeleton[p] = 1;
             else if (nsup==0) skeleton[p] = 2;
         }
         labelList = skeleton;
    }

}
