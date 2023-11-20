package nl.fullbrainpicture.libraries;

import java.io.*;
import java.util.*;

import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.utilities.*;

import org.apache.commons.math3.util.FastMath;

/**
 *
 *  This class computes various properties of surface meshes
 *	
 *  @version    Nov 2023
 *	@author     Pierre-Louis Bazin
 *		
 *
 */

public class MeshProcessing {
    
    private static final int X=0;
    private static final int Y=1;
    private static final int Z=2; 

    public static final int[] generateEdgePointList(int[] faceList) {
	    int nedges = faceList.length/2;
	    
	    int[] edgeList = new int[2*nedges];
	    int ne=0;
	    for (int f=0;f<faceList.length;f+=3) {
	        // only add the edges from low to high labels (arbitrary):
	        // that way edges are counted only once (triangles are oriented)
	        if (faceList[f+0]<faceList[f+1]) {
                edgeList[ne+0] =  faceList[f+0];
                edgeList[ne+1] =  faceList[f+1];
                ne += 2;
            }
            
	        if (faceList[f+1]<faceList[f+2]) {
                edgeList[ne+0] =  faceList[f+1];
                edgeList[ne+1] =  faceList[f+2];
                ne += 2;
            }
	        if (faceList[f+2]<faceList[f+0]) {
                edgeList[ne+0] =  faceList[f+2];
                edgeList[ne+1] =  faceList[f+0];
                ne += 2;
            }
        }
        return edgeList;
	}
	
	public static final int[] generateEdgeFaceList(int[] edgeList, int[][] pointFaceList) {
	    int nedges = edgeList.length;
	    
        int[] edgeFaceList = new int[2*nedges];
        for (int e=0;e<nedges;e++) {
            int p0 = edgeList[2*e+0];
            int p1 = edgeList[2*e+1];
            
            int de=0;
            for (int f0=0;f0<pointFaceList[p0].length;f0++) {
                for (int f1=0;f1<pointFaceList[p1].length;f1++) {
                    if (pointFaceList[p0][f0]==pointFaceList[p1][f1]) {
                        edgeFaceList[2*e+de] = pointFaceList[p0][f0];
                        de++;
                    }
                }
            }
            if (de!=2) System.out.print("!");
	    }
	    return edgeFaceList;   
	}
	
    public static final int[][] generatePointNeighborTable(int npt, int[] faces) {
        ArrayList[] table = new ArrayList[npt];
        for (int n=0;n<npt;n++) table[n] = new ArrayList<Integer>();
        for (int f=0;f<faces.length;f+=3) {
            table[faces[f+0]].add(faces[f+1]);
            //table[faces[f+0]].add(faces[f+2])
            
            table[faces[f+1]].add(faces[f+2]);
            //table[faces[f+1]].add(faces[f+0])
            
            table[faces[f+2]].add(faces[f+0]);
            //table[faces[f+2]].add(faces[f+1])
        }
        int[][] map = new int[npt][];
        for (int n=0;n<npt;n++) {
            map[n] = new int[table[n].size()];
            for (int m=0;m<table[n].size();m++)
                map[n][m] = (int)table[n].get(m);
        }
        return map;
    }
    
    public static final int[][] generateFaceNeighborTable(int npt, int[] faces) {
        ArrayList[] table = new ArrayList[npt];
        for (int n=0;n<npt;n++) table[n] = new ArrayList();
        for (int f=0;f<faces.length;f+=3) {
            table[faces[f+0]].add(f/3);
            table[faces[f+1]].add(f/3);
            table[faces[f+2]].add(f/3);
        }
        int[][] map = new int[npt][];
        for (int n=0;n<npt;n++) {
            map[n] = new int[table[n].size()];
            for (int m=0;m<table[n].size();m++)
                map[n][m] = (int)table[n].get(m);
        }
        return map;
    }
    
    public static final float[] getTriangleCenter(int id, float[] pts, int[] faces) {
        float[] center = new float[3];
        center[0] += pts[3*faces[3*id+0]+0]/3.0f;
        center[1] += pts[3*faces[3*id+0]+1]/3.0f;
        center[2] += pts[3*faces[3*id+0]+2]/3.0f;
        
        center[0] += pts[3*faces[3*id+1]+0]/3.0f;
        center[1] += pts[3*faces[3*id+1]+1]/3.0f;
        center[2] += pts[3*faces[3*id+1]+2]/3.0f;
        
        center[0] += pts[3*faces[3*id+2]+0]/3.0f;
        center[1] += pts[3*faces[3*id+2]+1]/3.0f;
        center[2] += pts[3*faces[3*id+2]+2]/3.0f;
        
        return center;
    }

    public static final float getTriangleArea(int id, float[] pts, int[] faces) {
        double vx, vy, vz;
        
        float V0x = pts[3*faces[3*id+0]+0];
        float V0y = pts[3*faces[3*id+0]+1];
        float V0z = pts[3*faces[3*id+0]+2];
        
        float V1x = pts[3*faces[3*id+1]+0];
        float V1y = pts[3*faces[3*id+1]+1];
        float V1z = pts[3*faces[3*id+1]+2];
        
        float V2x = pts[3*faces[3*id+2]+0];
        float V2y = pts[3*faces[3*id+2]+1];
        float V2z = pts[3*faces[3*id+2]+2];
        
        vx = ((V0y * V1z) - (V1y * V0z) + (V1y * V2z) - (V2y * V1z) + (V2y * V0z) - (V0y * V2z));
        vy = ((V0z * V1x) - (V1z * V0x) + (V1z * V2x) - (V2z * V1x) + (V2z * V0x) - (V0z * V2x));
        vz = ((V0x * V1y) - (V1x * V0y) + (V1x * V2y) - (V2x * V1y) + (V2x * V0y) - (V0x * V2y));
        
        return (float) (0.5 * Math.sqrt((vx * vx) + (vy * vy) + (vz * vz)));
    }
    
    public static final float[] computeInsideDistanceApproximation(float[] pts, int[] faces, int[] labels) {
        
        int npt = pts.length/3;
        
        int[][] ngb =  generatePointNeighborTable(npt, faces);
        
        BinaryHeapPair heap = new BinaryHeapPair(npt, BinaryHeapPair.MINTREE);
		
        // region growing from boundary inside
        for (int p=0;p<npt;p++) if (labels[p]>0) {
            // check neighborhhod
            boolean boundary=false;
            float mindist=1e9f;
            for (int k=0;k<ngb[p].length;k++) {
                int nk = ngb[p][k];
                if (labels[nk]!=labels[p]) {
                    boundary = true;
                    float dist = (pts[3*p+X]-pts[3*nk+X])*(pts[3*p+X]-pts[3*nk+X])
                                +(pts[3*p+Y]-pts[3*nk+Y])*(pts[3*p+Y]-pts[3*nk+Y])
                                +(pts[3*p+Z]-pts[3*nk+Z])*(pts[3*p+Z]-pts[3*nk+Z]);
                                
                    if (dist<mindist) mindist = dist;                            
                }
            }
            if (boundary) {
                mindist = 0.5f*(float)FastMath.sqrt(mindist);
                heap.addValue(mindist, p, labels[p]);
            }
        }
        
        boolean[] processed = new boolean[npt];
        float[] distance = new float[npt];
        
        while (heap.isNotEmpty()) {
             // extract point with minimum distance
        	float dist = heap.getFirst();
        	int p = heap.getFirstId1();
        	int lb = heap.getFirstId2();
			heap.removeFirst();

			// if more than nmgdm labels have been found already, this is done
			if (processed[p])  continue;
			
			// update the distance functions at the current level
			distance[p] = dist;
			processed[p]=true; // update the current level
 			
			// find new neighbors
			for (int k = 0; k<ngb[p].length; k++) {
				int nk = ngb[p][k];
				
				if (labels[nk]==labels[p] && !processed[nk]) {
				    // just use the minimum distance along edges (very rough approximation)
				    float mindist = 1e9f;
                    for (int l=0;l<ngb[nk].length;l++) {
				        int nl = ngb[nk][l];
				        
				        if (processed[nl]) {
                            float newdist = (pts[3*nk+X]-pts[3*nl+X])*(pts[3*nk+X]-pts[3*nl+X])
                                        +(pts[3*nk+Y]-pts[3*nl+Y])*(pts[3*nk+Y]-pts[3*nl+Y])
                                        +(pts[3*nk+Z]-pts[3*nl+Z])*(pts[3*nk+Z]-pts[3*nl+Z]);
                            if (newdist<mindist) mindist = newdist;        
                        }
                    }
                    heap.addValue(dist+(float)FastMath.sqrt(mindist), nk, lb);
                }
            }
                    
		}
     
        return distance;
    }
     
    public static final float[] computeInsideDistance(float[] pts, int[] faces, int[] labels) {
        
        int npt = pts.length/3;
        
        int[][] ngbp =  generatePointNeighborTable(npt, faces);
        int[][] ngbf =  generateFaceNeighborTable(npt, faces);
        
        BinaryHeapPair heap = new BinaryHeapPair(npt, BinaryHeapPair.MINTREE);
		
        // region growing from boundary inside
        for (int p=0;p<npt;p++) if (labels[p]>0) {
            // check neighborhhod
            boolean boundary=false;
            float mindist=1e9f;
            for (int k=0;k<ngbp[p].length;k++) {
                int nk = ngbp[p][k];
                if (labels[nk]!=labels[p]) {
                    boundary = true;
                    // look for smallest line distance among neighboring faces
                    for (int f=0;f<ngbf[p].length;f++) {
                        int nf = ngbf[p][f];
                        int p1=p, p2=p;
                        if (faces[3*nf+0]==p) { 
                            p1 = faces[3*nf+1];
                            p2 = faces[3*nf+2];
                        } else if (faces[3*nf+1]==p) {
                            p1 = faces[3*nf+2];
                            p2 = faces[3*nf+0];
                        } else if (faces[3*nf+2]==p) {
                            p1 = faces[3*nf+0];
                            p2 = faces[3*nf+1];
                        } else {
                            System.out.print("!");
                        }
                        if (labels[p1]!=labels[p] && labels[p2]!=labels[p]) {
                            // distance to line between the two points if inside, distance to closest otherwise
                            // d(P, L_P1P2) = || (P-P1) - (P-P1).(P2-P1) / || P2-P1 ||
                            double d1 = FastMath.sqrt( (pts[3*p+X]-pts[3*p1+X])*(pts[3*p+X]-pts[3*p1+X])
                                                      +(pts[3*p+Y]-pts[3*p1+Y])*(pts[3*p+Y]-pts[3*p1+Y])
                                                      +(pts[3*p+Z]-pts[3*p1+Z])*(pts[3*p+Z]-pts[3*p1+Z]) );
                            double d2 = FastMath.sqrt( (pts[3*p+X]-pts[3*p2+X])*(pts[3*p+X]-pts[3*p2+X])
                                                      +(pts[3*p+Y]-pts[3*p2+Y])*(pts[3*p+Y]-pts[3*p2+Y])
                                                      +(pts[3*p+Z]-pts[3*p2+Z])*(pts[3*p+Z]-pts[3*p2+Z]) );
                            double d12 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p2+X])*(pts[3*p1+X]-pts[3*p2+X])
                                                       +(pts[3*p1+Y]-pts[3*p2+Y])*(pts[3*p1+Y]-pts[3*p2+Y])
                                                       +(pts[3*p1+Z]-pts[3*p2+Z])*(pts[3*p1+Z]-pts[3*p2+Z]) );
                            double d0 = FastMath.sqrt( Numerics.square( (pts[3*p+X]-pts[3*p1+X])*(1.0f-(pts[3*p2+X]-pts[3*p1+X])/d12) )
                                                      +Numerics.square( (pts[3*p+Y]-pts[3*p1+Y])*(1.0f-(pts[3*p2+Y]-pts[3*p1+Y])/d12) )
                                                      +Numerics.square( (pts[3*p+Z]-pts[3*p1+Z])*(1.0f-(pts[3*p2+Z]-pts[3*p1+Z])/d12) ) );
                            
                            mindist = Numerics.min(mindist, 0.5f*(float)d0,0.5f*(float)d1,0.5f*(float)d2);    
                        } else if  (labels[p1]!=labels[p]) {
                            // distance P1 to P, P2
                            double d1 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p+X])*(pts[3*p1+X]-pts[3*p+X])
                                                      +(pts[3*p1+Y]-pts[3*p+Y])*(pts[3*p1+Y]-pts[3*p+Y])
                                                      +(pts[3*p1+Z]-pts[3*p+Z])*(pts[3*p1+Z]-pts[3*p+Z]) );
                            double d2 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p2+X])*(pts[3*p1+X]-pts[3*p2+X])
                                                      +(pts[3*p1+Y]-pts[3*p2+Y])*(pts[3*p1+Y]-pts[3*p2+Y])
                                                      +(pts[3*p1+Z]-pts[3*p2+Z])*(pts[3*p1+Z]-pts[3*p2+Z]) );
                            double d12 = FastMath.sqrt( (pts[3*p+X]-pts[3*p2+X])*(pts[3*p+X]-pts[3*p2+X])
                                                       +(pts[3*p+Y]-pts[3*p2+Y])*(pts[3*p+Y]-pts[3*p2+Y])
                                                       +(pts[3*p+Z]-pts[3*p2+Z])*(pts[3*p+Z]-pts[3*p2+Z]) );
                            double d0 = FastMath.sqrt( Numerics.square( (pts[3*p1+X]-pts[3*p+X])*(1.0f-(pts[3*p2+X]-pts[3*p+X])/d12) )
                                                      +Numerics.square( (pts[3*p1+Y]-pts[3*p+Y])*(1.0f-(pts[3*p2+Y]-pts[3*p+Y])/d12) )
                                                      +Numerics.square( (pts[3*p1+Z]-pts[3*p+Z])*(1.0f-(pts[3*p2+Z]-pts[3*p+Z])/d12) ) );
                            
                            mindist = Numerics.min(mindist, 0.5f*(float)d0,0.5f*(float)d1,0.5f*(float)d2);    
                        } else if  (labels[p2]!=labels[p]) {
                            // distance P2 to P, P1
                            double d1 = FastMath.sqrt( (pts[3*p2+X]-pts[3*p+X])*(pts[3*p2+X]-pts[3*p+X])
                                                      +(pts[3*p2+Y]-pts[3*p+Y])*(pts[3*p2+Y]-pts[3*p+Y])
                                                      +(pts[3*p2+Z]-pts[3*p+Z])*(pts[3*p2+Z]-pts[3*p+Z]) );
                            double d2 = FastMath.sqrt( (pts[3*p2+X]-pts[3*p1+X])*(pts[3*p2+X]-pts[3*p1+X])
                                                      +(pts[3*p2+Y]-pts[3*p1+Y])*(pts[3*p2+Y]-pts[3*p1+Y])
                                                      +(pts[3*p2+Z]-pts[3*p1+Z])*(pts[3*p2+Z]-pts[3*p1+Z]) );
                            double d12 = FastMath.sqrt( (pts[3*p+X]-pts[3*p1+X])*(pts[3*p+X]-pts[3*p1+X])
                                                       +(pts[3*p+Y]-pts[3*p1+Y])*(pts[3*p+Y]-pts[3*p1+Y])
                                                       +(pts[3*p+Z]-pts[3*p1+Z])*(pts[3*p+Z]-pts[3*p1+Z]) );
                            double d0 = FastMath.sqrt( Numerics.square( (pts[3*p2+X]-pts[3*p+X])*(1.0f-(pts[3*p1+X]-pts[3*p+X])/d12) )
                                                      +Numerics.square( (pts[3*p2+Y]-pts[3*p+Y])*(1.0f-(pts[3*p1+Y]-pts[3*p+Y])/d12) )
                                                      +Numerics.square( (pts[3*p2+Z]-pts[3*p+Z])*(1.0f-(pts[3*p1+Z]-pts[3*p+Z])/d12) ) );
                            
                            mindist = Numerics.min(mindist, 0.5f*(float)d0,0.5f*(float)d1,0.5f*(float)d2);    
                        }
                    }
                }
            }
            if (boundary) {
                heap.addValue(mindist, p, labels[p]);
            }
        }
        
        boolean[] processed = new boolean[npt];
        float[] distance = new float[npt];
        
        while (heap.isNotEmpty()) {
             // extract point with minimum distance
        	float dist = heap.getFirst();
        	int p = heap.getFirstId1();
        	int lb = heap.getFirstId2();
			heap.removeFirst();

			// if more than nmgdm labels have been found already, this is done
			if (processed[p])  continue;
			
			// update the distance functions at the current level
			distance[p] = dist;
			processed[p]=true; // update the current level
 			
			// find new neighbors
			for (int k = 0; k<ngbp[p].length; k++) {
				int nk = ngbp[p][k];
				
				if (labels[nk]==lb && !processed[nk]) {
				    float mindist=1e9f;
                    // look for smallest line distance among neighboring faces
                    for (int f=0;f<ngbf[nk].length;f++) {
                        int nf = ngbf[nk][f];
                        int p1=nk, p2=nk;
                        if (faces[3*nf+0]==nk) { 
                            p1 = faces[3*nf+1];
                            p2 = faces[3*nf+2];
                        } else if (faces[3*nf+1]==nk) {
                            p1 = faces[3*nf+2];
                            p2 = faces[3*nf+0];
                        } else if (faces[3*nf+2]==nk) {
                            p1 = faces[3*nf+0];
                            p2 = faces[3*nf+1];
                        } else {
                            System.out.print("!");
                        }
                        if (processed[p1] && processed[p2]) {                            
                            // distance to line between the two points if inside, distance to closest otherwise
                            // d(P, L_P1P2) = || (P-P1) - (P-P1).(P2-P1) / || P2-P1 ||
                            
                            // note that it is still an approximation, just a better one
                            double d1 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p1+X])*(pts[3*nk+X]-pts[3*p1+X])
                                                      +(pts[3*nk+Y]-pts[3*p1+Y])*(pts[3*nk+Y]-pts[3*p1+Y])
                                                      +(pts[3*nk+Z]-pts[3*p1+Z])*(pts[3*nk+Z]-pts[3*p1+Z]) );
                            double d2 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p2+X])*(pts[3*nk+X]-pts[3*p2+X])
                                                      +(pts[3*nk+Y]-pts[3*p2+Y])*(pts[3*nk+Y]-pts[3*p2+Y])
                                                      +(pts[3*nk+Z]-pts[3*p2+Z])*(pts[3*nk+Z]-pts[3*p2+Z]) );
                            double d12 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p2+X])*(pts[3*p1+X]-pts[3*p2+X])
                                                       +(pts[3*p1+Y]-pts[3*p2+Y])*(pts[3*p1+Y]-pts[3*p2+Y])
                                                       +(pts[3*p1+Z]-pts[3*p2+Z])*(pts[3*p1+Z]-pts[3*p2+Z]) );
                            double d0 = FastMath.sqrt( Numerics.square( (pts[3*nk+X]-pts[3*p1+X])*(1.0f-(pts[3*p2+X]-pts[3*p1+X])/d12) )
                                                      +Numerics.square( (pts[3*nk+Y]-pts[3*p1+Y])*(1.0f-(pts[3*p2+Y]-pts[3*p1+Y])/d12) )
                                                      +Numerics.square( (pts[3*nk+Z]-pts[3*p1+Z])*(1.0f-(pts[3*p2+Z]-pts[3*p1+Z])/d12) ) );
                        
                            mindist = Numerics.min(mindist, 0.5f*(distance[p1]+distance[p2])+(float)d0,
                                                   distance[p1]+(float)d1, distance[p2]+(float)d2);
                        } else if (processed[p1]) {
                            double d1 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p1+X])*(pts[3*nk+X]-pts[3*p1+X])
                                                      +(pts[3*nk+Y]-pts[3*p1+Y])*(pts[3*nk+Y]-pts[3*p1+Y])
                                                      +(pts[3*nk+Z]-pts[3*p1+Z])*(pts[3*nk+Z]-pts[3*p1+Z]) );

                            mindist = Numerics.min(mindist, distance[p1]+(float)d1);
                        } else if (processed[p2]) {
                            double d2 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p2+X])*(pts[3*nk+X]-pts[3*p2+X])
                                                      +(pts[3*nk+Y]-pts[3*p2+Y])*(pts[3*nk+Y]-pts[3*p2+Y])
                                                      +(pts[3*nk+Z]-pts[3*p2+Z])*(pts[3*nk+Z]-pts[3*p2+Z]) );

                            mindist = Numerics.min(mindist, distance[p2]+(float)d2);
                        }
                    }
                    
                    heap.addValue(mindist, nk, lb);
                }
            }
                    
		}
        return distance;
    }
   
    public static final void computeOutsideDistanceFunctions(int nb, float[][] distances, int[][] closest, float[] pts, int[] faces, int[] labels) {
        
        // assume we have nb x npt arrays for distances, closest
        
        // here we only propagate labels > 0
        
        int npt = pts.length/3;
        int nfc = faces.length/3;
        
        int[][] ngbf =  generateFaceNeighborTable(npt, faces);
        int[][] ngbp =  generatePointNeighborTable(npt, faces);
        
        BinaryHeapPair heap = new BinaryHeapPair(npt, BinaryHeapPair.MINTREE);
		
        // region growing from boundary inside
        for (int nf=0;nf<nfc;nf++) {
            // check by faces, not points
            if (labels[faces[3*nf+0]]!=labels[faces[3*nf+1]] 
                || labels[faces[3*nf+1]]!=labels[faces[3*nf+2]] 
                    || labels[faces[3*nf+2]]!=labels[faces[3*nf+0]]) {

                // either two or three labels
                if (labels[faces[3*nf+0]]!=labels[faces[3*nf+1]] 
                    && labels[faces[3*nf+1]]!=labels[faces[3*nf+2]] 
                        && labels[faces[3*nf+2]]!=labels[faces[3*nf+0]]) {
                
                    // three label junction
                    double d01 = FastMath.sqrt( Numerics.square(pts[3*faces[3*nf+0]+X]-pts[3*faces[3*nf+1]+X])
                                               +Numerics.square(pts[3*faces[3*nf+0]+Y]-pts[3*faces[3*nf+1]+Y])
                                               +Numerics.square(pts[3*faces[3*nf+0]+Z]-pts[3*faces[3*nf+1]+Z]) );
                    double d12 = FastMath.sqrt( Numerics.square(pts[3*faces[3*nf+1]+X]-pts[3*faces[3*nf+2]+X])
                                               +Numerics.square(pts[3*faces[3*nf+1]+Y]-pts[3*faces[3*nf+2]+Y])
                                               +Numerics.square(pts[3*faces[3*nf+1]+Z]-pts[3*faces[3*nf+2]+Z]) );
                    double d20 = FastMath.sqrt( Numerics.square(pts[3*faces[3*nf+2]+X]-pts[3*faces[3*nf+0]+X])
                                               +Numerics.square(pts[3*faces[3*nf+2]+Y]-pts[3*faces[3*nf+0]+Y])
                                               +Numerics.square(pts[3*faces[3*nf+2]+Z]-pts[3*faces[3*nf+0]+Z]) );
                    
                    // add all possible distances
                    if (labels[faces[3*nf+1]]>0 && labels[faces[3*nf+0]]>-1) heap.addValue(0.5f*(float)d01, faces[3*nf+0], labels[faces[3*nf+1]]);
                    if (labels[faces[3*nf+0]]>0 && labels[faces[3*nf+1]]>-1) heap.addValue(0.5f*(float)d01, faces[3*nf+1], labels[faces[3*nf+0]]);
                    if (labels[faces[3*nf+2]]>0 && labels[faces[3*nf+1]]>-1) heap.addValue(0.5f*(float)d12, faces[3*nf+1], labels[faces[3*nf+2]]);
                    if (labels[faces[3*nf+1]]>0 && labels[faces[3*nf+2]]>-1) heap.addValue(0.5f*(float)d12, faces[3*nf+2], labels[faces[3*nf+1]]);
                    if (labels[faces[3*nf+0]]>0 && labels[faces[3*nf+2]]>-1) heap.addValue(0.5f*(float)d20, faces[3*nf+2], labels[faces[3*nf+0]]);
                    if (labels[faces[3*nf+2]]>0 && labels[faces[3*nf+0]]>-1) heap.addValue(0.5f*(float)d20, faces[3*nf+0], labels[faces[3*nf+2]]);
                } else {
                    
                    // find the one that is different
                    int p=faces[3*nf+0], p1=faces[3*nf+1], p2=faces[3*nf+2];
                    if (labels[p]==labels[p1]) {
                        int swap = p2;
                        p2 = p;
                        p = swap;
                    } else if (labels[p]==labels[p2]) {
                        int swap = p1;
                        p1 = p;
                        p = swap;
                    }
                    // distance to line between the two points if inside, distance to closest otherwise
                    // d(P, L_P1P2) = || (P-P1) - (P-P1).(P2-P1) / || P2-P1 ||
                    double d1 = FastMath.sqrt( (pts[3*p+X]-pts[3*p1+X])*(pts[3*p+X]-pts[3*p1+X])
                                              +(pts[3*p+Y]-pts[3*p1+Y])*(pts[3*p+Y]-pts[3*p1+Y])
                                              +(pts[3*p+Z]-pts[3*p1+Z])*(pts[3*p+Z]-pts[3*p1+Z]) );
                    double d2 = FastMath.sqrt( (pts[3*p+X]-pts[3*p2+X])*(pts[3*p+X]-pts[3*p2+X])
                                              +(pts[3*p+Y]-pts[3*p2+Y])*(pts[3*p+Y]-pts[3*p2+Y])
                                              +(pts[3*p+Z]-pts[3*p2+Z])*(pts[3*p+Z]-pts[3*p2+Z]) );
                    double d12 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p2+X])*(pts[3*p1+X]-pts[3*p2+X])
                                               +(pts[3*p1+Y]-pts[3*p2+Y])*(pts[3*p1+Y]-pts[3*p2+Y])
                                               +(pts[3*p1+Z]-pts[3*p2+Z])*(pts[3*p1+Z]-pts[3*p2+Z]) );
                    double d0 = FastMath.sqrt( Numerics.square( (pts[3*p+X]-pts[3*p1+X])*(1.0f-(pts[3*p2+X]-pts[3*p1+X])/d12) )
                                              +Numerics.square( (pts[3*p+Y]-pts[3*p1+Y])*(1.0f-(pts[3*p2+Y]-pts[3*p1+Y])/d12) )
                                              +Numerics.square( (pts[3*p+Z]-pts[3*p1+Z])*(1.0f-(pts[3*p2+Z]-pts[3*p1+Z])/d12) ) );
                    
                    float dist = Numerics.min(0.5f*(float)d0,0.5f*(float)d1,0.5f*(float)d2);
                    
                    if (labels[p1]>0 && labels[p]>-1) heap.addValue(dist, p, labels[p1]);
                    if (labels[p]>0 && labels[p1]>-1) heap.addValue(dist, p1, labels[p]);
                    if (labels[p]>0 && labels[p2]>-1) heap.addValue(dist, p2, labels[p]);
                }
             }
         }
        
        int[] processed = new int[npt];
        while (heap.isNotEmpty()) {
             // extract point with minimum distance
        	float dist = heap.getFirst();
        	int p = heap.getFirstId1();
        	int lb = heap.getFirstId2();
			heap.removeFirst();

			// if more than nb labels have been found already, this is done
			if (processed[p]>=nb)  continue;
			
			// check if the current label is already accounted for
			boolean found=false;
			for (int n=0;n<processed[p];n++) if (closest[n][p]==lb) found=true;
			if (found) continue;
			
			// update the distance functions at the current level
			distances[processed[p]][p] = dist;
			closest[processed[p]][p] = lb;
			processed[p]++; // update the current level
 			
			// find new neighbors
			for (int k = 0; k<ngbp[p].length; k++) {
				int nk = ngbp[p][k];
				
                if (labels[nk]!=lb && labels[nk]>-1) {
                    found=false;
                    for (int n=0;n<processed[nk];n++) if (closest[n][nk]==lb) found=true;
                    
                    if (!found) {
                        float mindist=1e9f;
                        // look for smallest line distance among neighboring faces
                        for (int f=0;f<ngbf[nk].length;f++) {
                            int nf = ngbf[nk][f];
                            int p1=nk, p2=nk;
                            if (faces[3*nf+0]==nk) { 
                                p1 = faces[3*nf+1];
                                p2 = faces[3*nf+2];
                            } else if (faces[3*nf+1]==nk) {
                                p1 = faces[3*nf+2];
                                p2 = faces[3*nf+0];
                            } else if (faces[3*nf+2]==nk) {
                                p1 = faces[3*nf+0];
                                p2 = faces[3*nf+1];
                            } else {
                                System.out.print("!");
                            }
                            
                            int found1=-1;
                            for (int n=0;n<processed[p1];n++) if (closest[n][p1]==lb) found1=n;
                    
                            int found2=-1;
                            for (int n=0;n<processed[p2];n++) if (closest[n][p2]==lb) found2=n;
                    
                            if (found1>-1 && found2>-1) {                            
                                // distance to line between the two points if inside, distance to closest otherwise
                                // d(P, L_P1P2) = || (P-P1) - (P-P1).(P2-P1) / || P2-P1 ||
                                
                                // note that it is still an approximation, just a better one
                                double d1 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p1+X])*(pts[3*nk+X]-pts[3*p1+X])
                                                          +(pts[3*nk+Y]-pts[3*p1+Y])*(pts[3*nk+Y]-pts[3*p1+Y])
                                                          +(pts[3*nk+Z]-pts[3*p1+Z])*(pts[3*nk+Z]-pts[3*p1+Z]) );
                                double d2 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p2+X])*(pts[3*nk+X]-pts[3*p2+X])
                                                          +(pts[3*nk+Y]-pts[3*p2+Y])*(pts[3*nk+Y]-pts[3*p2+Y])
                                                          +(pts[3*nk+Z]-pts[3*p2+Z])*(pts[3*nk+Z]-pts[3*p2+Z]) );
                                double d12 = FastMath.sqrt( (pts[3*p1+X]-pts[3*p2+X])*(pts[3*p1+X]-pts[3*p2+X])
                                                           +(pts[3*p1+Y]-pts[3*p2+Y])*(pts[3*p1+Y]-pts[3*p2+Y])
                                                           +(pts[3*p1+Z]-pts[3*p2+Z])*(pts[3*p1+Z]-pts[3*p2+Z]) );
                                double d0 = FastMath.sqrt( Numerics.square( (pts[3*nk+X]-pts[3*p1+X])*(1.0f-(pts[3*p2+X]-pts[3*p1+X])/d12) )
                                                          +Numerics.square( (pts[3*nk+Y]-pts[3*p1+Y])*(1.0f-(pts[3*p2+Y]-pts[3*p1+Y])/d12) )
                                                          +Numerics.square( (pts[3*nk+Z]-pts[3*p1+Z])*(1.0f-(pts[3*p2+Z]-pts[3*p1+Z])/d12) ) );
                            
                                mindist = Numerics.min(mindist, 0.5f*(distances[found1][p1]+distances[found2][p2])+(float)d0,
                                                       distances[found1][p1]+(float)d1, distances[found2][p2]+(float)d2);
                            } else if (found1>-1) {
                                double d1 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p1+X])*(pts[3*nk+X]-pts[3*p1+X])
                                                          +(pts[3*nk+Y]-pts[3*p1+Y])*(pts[3*nk+Y]-pts[3*p1+Y])
                                                          +(pts[3*nk+Z]-pts[3*p1+Z])*(pts[3*nk+Z]-pts[3*p1+Z]) );
    
                                mindist = Numerics.min(mindist, distances[found1][p1]+(float)d1);
                            } else if (found2>-1) {
                                double d2 = FastMath.sqrt( (pts[3*nk+X]-pts[3*p2+X])*(pts[3*nk+X]-pts[3*p2+X])
                                                          +(pts[3*nk+Y]-pts[3*p2+Y])*(pts[3*nk+Y]-pts[3*p2+Y])
                                                          +(pts[3*nk+Z]-pts[3*p2+Z])*(pts[3*nk+Z]-pts[3*p2+Z]) );
    
                                mindist = Numerics.min(mindist, distances[found2][p2]+(float)d2);
                            }
                        }
                        heap.addValue(mindist, nk, lb);
                    }
                }
            }
                    
		}
        return;
        
    }
    
}
