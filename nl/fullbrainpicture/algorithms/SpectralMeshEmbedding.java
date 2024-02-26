package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.linear.*;
//import Jama.*;
//import org.jblas.*;

import java.util.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SpectralMeshEmbedding {

    
	// jist containers
    private float[] pointList;
    private int[] triangleList;
    private float[] embeddingList;
    
    private float[] pointListRef;
    private int[] triangleListRef;
    private float[] embeddingListRef;

	private int ndims = 10;
	private int msize = 800;
	private float scale = 1.0f;
	private float link = 1.0f;
	
	// numerical quantities
	private static final	double	INVSQRT2 = 1.0/FastMath.sqrt(2.0);
	private static final	double	INVSQRT3 = 1.0/FastMath.sqrt(3.0);
	private static final	double	SQRT2 = FastMath.sqrt(2.0);
	private static final	double	SQRT3 = FastMath.sqrt(3.0);

	// direction labeling		
	public	static	byte	X = 0;
	public	static	byte	Y = 1;
	public	static	byte	Z = 2;
	public	static	byte	T = 3;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setSurfacePoints(float[] val) { pointList = val; }
	public final void setSurfaceTriangles(int[] val) { triangleList = val; }

	public final void setReferencePoints(float[] val) { pointListRef = val; }
	public final void setReferenceTriangles(int[] val) { triangleListRef = val; }


	public final void setDimensions(int val) { ndims = val; }
	public final void setMatrixSize(int val) { msize = val; }
	public final void setDistanceScale(float val) { scale = val; }
	public final void setLinkingFactor(float val) { link = val; }
					
	// create outputs
	public final float[] 	getEmbeddingValues() { return embeddingList; }
	public final float[] 	getReferenceEmbeddingValues() { return embeddingListRef; }
	
	public void meshDistanceEmbedding() {
	    
	    // data size
	    int npt = pointList.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
        
	    // precompute surface-based distances
	    int[][] ngbp =  MeshProcessing.generatePointNeighborTable(npt, triangleList);
	    float avgp = 0.0f;
	    int maxp = 0;
        for (int n=0;n<npt;n++) {
            avgp += ngbp[n].length;
            if (ngbp[n].length>maxp) maxp = ngbp[n].length;
        }
        avgp /= npt;
        System.out.println("Average connectivity: "+avgp+" (max: "+maxp+")");
	    
	    int depth = Numerics.ceil(avgp);
	    int[] sampleList = new int[npt];
	    for (int n=0;n<msize*step;n+=step) {
	        sampleList[n] = n/step+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, sampleList);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<npt;n++) {
            if (distances[d][n]>maxdist) maxdist = distances[d][n];
        }
	    System.out.println("fast marching distances max: "+maxdist);
	    
        // build a complete sample distance map? should be doable, roughly O(msize^2)
        float[][] sampledist = new float[msize][msize];
        for (int n=0;n<msize*step;n+=step) {
            for (int d=0;d<depth;d++) {
	            int m = (closest[d][n]-1)*step;
	            if (m>=0) {
	                sampledist[n/step][m/step] = distances[d][n];
	                sampledist[m/step][n/step] = distances[d][n];
	            }
	        }
	    }
        float dmax=0.0f, dmean=0.0f;
        int nmean=0;
        for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
            if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
            if (sampledist[n/step][m/step]>0) {
                dmean += sampledist[n/step][m/step];
                nmean++;
            }
        }
        dmean /= nmean;
        System.out.println("(mean: "+dmean+", max:"+dmax+")");

	    // set to false to skip the propagation
	    boolean missing=true;
	    int nmiss=0;
	    while (missing) {
            missing=false;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>0) {
                    for (int d=0;d<depth;d++) {
                        int jm = (closest[d][n]-1)*step;
                        if (jm>=0) {
                            if (sampledist[jm/step][m/step]==0) sampledist[jm/step][m/step] = distances[d][n]+sampledist[n/step][m/step];
                            else sampledist[jm/step][m/step] = Numerics.min(sampledist[jm/step][m/step],distances[d][n]+sampledist[n/step][m/step]);
                            sampledist[m/step][jm/step] = sampledist[jm/step][m/step];
                        }
                        int jn = (closest[d][m]-1)*step;
                        if (jn>=0) {
                            if (sampledist[jn/step][n/step]==0) sampledist[jn/step][n/step] = distances[d][m]+sampledist[n/step][m/step];
                            else sampledist[jn/step][n/step] = Numerics.min(sampledist[jn/step][n/step],distances[d][m]+sampledist[n/step][m/step]);
                            sampledist[n/step][jn/step] = sampledist[jn/step][n/step];
                        }
                    }
                } else {
                    missing=true;
                }
            }
            nmiss++;
        }
        System.out.println("approximate distance propagation: "+nmiss);
	    dmax=0.0f; 
	    dmean=0.0f;
        for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
            if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
            dmean += sampledist[n/step][m/step];
        }
        dmean /= msize*msize;
        System.out.println("(mean: "+dmean+", max:"+dmax+")");
        
        // reset diagonal to zero to have correct distance when closest
	    for (int n=0;n<msize*step;n+=step) {
	        sampledist[n/step][n/step] = 0.0f;
	    }
        
        if (scale<0) scale = dmean;
        
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
	        // self affinitiy should be 1?
            Azero[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {	            
                double dist = sampledist[n/step][m/step];
                
                if (dist>0) {
                    Azero[n/step][m/step] = affinity(dist);
                    Azero[m/step][n/step] = Azero[n/step][m/step];
                }
            }
        }
	    // First decomposition for degree: A
        RealMatrix mtx = new Array2DRowRealMatrix(Azero);
        EigenDecomposition eig = new EigenDecomposition(mtx);

        double[][] Ainv = new double[msize][msize];
	    for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            Ainv[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                Ainv[n][m] += eig.getV().getEntry(n,p)
                                *1.0/eig.getRealEigenvalue(p)
                                *eig.getV().getEntry(m,p);
            }
        }
        mtx = null;
        eig = null;

	    double[] a0r = new double[msize];
	    double[] b0r = new double[msize];
	    
	    // first the rows from A, B^T
	    for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {	    
	            a0r[n/step] += Azero[n/step][m/step];
	        }
	        for (int m=0;m<npt;m++) if (m%step!=0 || m>=msize*step) {
                double dist = distances[0][m]+sampledist[n/step][closest[0][m]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][m]>0) {
                    dist = Numerics.min(dist, distances[d][m]+sampledist[n/step][closest[d][m]-1]);
                }
                b0r[n/step] += affinity(dist);
            }
	    }
	    // convolve rows of B^T with A^-1
	    double[] ainvb0r = new double[msize];
	    for (int n=0;n<msize;n++) {
	        ainvb0r[n] = 0.0;
	        for (int m=0;m<msize;m++) {
	            ainvb0r[n] += Ainv[n][m]*b0r[m];
	        }
	    }
	    
	    // finally the degree
	    double[] degree = new double[npt];
	    for (int n=0;n<msize*step;n+=step) {
	        degree[n] = a0r[n/step]+b0r[n/step];
	    }
        for (int n=0;n<npt;n++) if (n%step!=0 || n>=msize*step) {
            for (int m=0;m<msize*step;m+=step) {	
                double dist = distances[0][n]+sampledist[m/step][closest[0][n]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][n]>0) {
                    dist = Numerics.min(dist, distances[d][n]+sampledist[m/step][closest[d][n]-1]);
                }
                    
                degree[n] += (1.0 + ainvb0r[m/step])*affinity(dist);
            }
        }
        System.out.println("build first approximation");
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
            Acore[n/step][n/step] = 1.0;
	        /* which one?
	        for (int m=n+step;m<msize*step;m+=step) {	            
                Acore[n/step][m/step] = -Azero[n/step][m/step]/degree[n];
                Acore[m/step][n/step] = -Azero[m/step][n/step]/degree[m];
            }*/
	        for (int m=n+step;m<msize*step;m+=step) {	            
                Acore[n/step][m/step] = -Azero[n/step][m/step]/FastMath.sqrt(degree[n]*degree[m]);
                Acore[m/step][n/step] = -Azero[m/step][n/step]/FastMath.sqrt(degree[n]*degree[m]);
            }
        }
	    // First decomposition: A
        mtx = new Array2DRowRealMatrix(Acore);
        eig = new EigenDecomposition(mtx);
        
        // sort eigenvalues: not needed, already done :)
        System.out.println("build orthogonalization");
        
        // build S = A + A^-1/2 BB^T A^-1/2
        double[][] sqAinv = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            sqAinv[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                sqAinv[n][m] += eig.getV().getEntry(n,p)
                                *1.0/FastMath.sqrt(eig.getRealEigenvalue(p))
                                *eig.getV().getEntry(m,p);
            }
        }
        mtx = null;
        eig = null;

        System.out.print(".");        

        // banded matrix is too big, we only compute BB^T
        double[][] BBt = new double[msize][msize];
	    // faster alternative?
        for (int j=0;j<npt;j++) if (j%step!=0 || j>=msize*step) {
            for (int n=0;n<msize*step;n+=step) {
                double distN = distances[0][j]+sampledist[n/step][closest[0][j]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][j]>0) {
                    distN = Numerics.min(distN, distances[d][j]+sampledist[n/step][closest[d][j]-1]);
                }   
                BBt[n/step][n/step] += affinity(distN)/degree[j]
                                      *affinity(distN)/degree[j];
                
                for (int m=n+step;m<msize*step;m+=step) {
                    double distM = distances[0][j]+sampledist[m/step][closest[0][j]-1];
                
                    for (int d=1;d<depth;d++) if (closest[d][j]>0) {
                        distM = Numerics.min(distM, distances[d][j]+sampledist[m/step][closest[d][j]-1]);
                    }
                    BBt[n/step][m/step] += affinity(distN)/degree[j]
                                          *affinity(distM)/degree[j];
                                  
                    BBt[m/step][n/step] += affinity(distN)/degree[j]
                                          *affinity(distM)/degree[j];
                }
            }
        }
	    System.out.print("..");        
        	    
        // update banded matrix
        double[][] sqAinvBBt = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            sqAinvBBt[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                sqAinvBBt[n][m] += sqAinv[n][p]*BBt[p][m];
            }
        }
        BBt = null;
        
	    System.out.print("...");        

        double[][] Afull = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            Afull[n][m] = Acore[n][m];
            for (int p=0;p<msize;p++) {
                Afull[n][m] += sqAinvBBt[n][p]*sqAinv[p][m];
            }
        }
        sqAinvBBt = null;

        System.out.print("....");        

        // second eigendecomposition: S = A + A^-1/2 BB^T A^-1/2
        mtx = new Array2DRowRealMatrix(Afull);
        eig = new EigenDecomposition(mtx);
        
        System.out.println("\nexport result to maps");

        // final result A^-1/2 V D^-1/2
        double[][] Vortho = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            Vortho[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                Vortho[n][m] += sqAinv[n][p]
                                *eig.getV().getEntry(p,m)
                                *1.0/FastMath.sqrt(eig.getRealEigenvalue(m));
            }
        }
        double[] evals = new double[ndims];
        for (int d=0;d<ndims;d++) evals[d] = eig.getRealEigenvalue(d);
        mtx = null;
        eig = null;
        
        // pass on to global result
        embeddingList = new float[npt*ndims];
        
        for (int dim=0;dim<ndims;dim++) {
            System.out.println("eigenvalue "+(dim+1)+": "+evals[dim]);
            for (int n=0;n<npt;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist = distances[0][n]+sampledist[m/step][closest[0][n]-1];
                
                    for (int d=1;d<depth;d++) if (closest[d][n]>0) {
                        dist = Numerics.min(dist, distances[d][n]+sampledist[m/step][closest[d][n]-1]);
                    }
                    
                    embed += affinity(dist)/degree[n]*Vortho[m/step][dim];
                }
                embeddingList[n+dim*npt] = (float)embed;
            }
        }
        
		return;
	}

	private final double affinity(double dist) {
	    //return scale/dist;
	    return 1.0/(1.0+dist/scale);
	    //return 1.0/(1.0+dist*dist/(scale*scale));
	}

	private final double linking(double dist) {
	    //return scale/dist;
	    return link/(1.0+dist/scale);
	    //return 1.0/(1.0+dist*dist/(scale*scale));
	}

	public final void meshDistanceSparseEmbedding(int depth, boolean eigenGame, boolean fullDistance, double alpha) {
	    //boolean eigenGame = true;
	    //boolean fullDistance = false;
	    //double alpha = 0.0;
	    
	    int npt=pointList.length/3;
            
	    // add a dimension for lowest eigenvector
	    //ndims = ndims+1;
	    
        // if volume is too big, subsample
        int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
            
	    // precompute surface-based distances
	    int[][] ngbp =  MeshProcessing.generatePointNeighborTable(npt, triangleList);
	    float avgp = 0.0f;
	    int maxp = 0;
        for (int n=0;n<npt;n++) {
            avgp += ngbp[n].length;
            if (ngbp[n].length>maxp) maxp = ngbp[n].length;
        }
        avgp /= npt;
        System.out.println("Average connectivity: "+avgp+" (max: "+maxp+")");
	    
        int nconnect = Numerics.ceil(avgp);
	    if (depth<0) depth = Numerics.ceil(avgp);
	    int[] sampleList = new int[npt];
	    for (int n=0;n<msize*step;n+=step) {
	        sampleList[n] = n/step+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, sampleList);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<npt;n++) {
            if (distances[d][n]>maxdist) maxdist = distances[d][n];
        }
	    System.out.println("fast marching distances max: "+maxdist);

	    double[][] matrix = new double[msize][msize];
	    
	    if (fullDistance) {
            // build a complete sample distance map? should be doable, roughly O(msize^2)
            // very slow for large meshes, though, and maybe not needed anyway given the eigengame step
            float[][] sampledist = new float[msize][msize];
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        sampledist[n/step][m/step] = distances[d][n];
                        sampledist[m/step][n/step] = distances[d][n];
                    }
                }
            }
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                if (sampledist[n/step][m/step]>0) {
                    dmean += sampledist[n/step][m/step];
                    nmean++;
                }
            }
            dmean /= nmean;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
    
            // set to false to skip the propagation
            int missing=1;
            int prev = -1;
            int nmiss=0;
            while (missing>0 && missing!=prev) {
                prev = missing;
                missing=0;
                for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                    if (sampledist[n/step][m/step]>0) {
                        for (int d=0;d<depth;d++) {
                            int jm = (closest[d][n]-1)*step;
                            if (jm>=0) {
                                if (sampledist[jm/step][m/step]==0) sampledist[jm/step][m/step] = distances[d][n]+sampledist[n/step][m/step];
                                else sampledist[jm/step][m/step] = Numerics.min(sampledist[jm/step][m/step],distances[d][n]+sampledist[n/step][m/step]);
                                sampledist[m/step][jm/step] = sampledist[jm/step][m/step];
                            }
                            int jn = (closest[d][m]-1)*step;
                            if (jn>=0) {
                                if (sampledist[jn/step][n/step]==0) sampledist[jn/step][n/step] = distances[d][m]+sampledist[n/step][m/step];
                                else sampledist[jn/step][n/step] = Numerics.min(sampledist[jn/step][n/step],distances[d][m]+sampledist[n/step][m/step]);
                                sampledist[n/step][jn/step] = sampledist[jn/step][n/step];
                            }
                        }
                    } else {
                        missing++;
                    }
                }
                nmiss++;
            }
            System.out.println("approximate distance propagation: "+nmiss);
            dmax=0.0f; 
            dmean=0.0f;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                dmean += sampledist[n/step][m/step];
            }
            dmean /= msize*msize;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
            
            // reset diagonal to zero to have correct distance when closest
            for (int n=0;n<msize*step;n+=step) {
                sampledist[n/step][n/step] = 0.0f;
            }
            
            if (scale<0) scale = dmean;
            
            for (int n=0;n<msize*step;n+=step) {
                for (int m=n+step;m<msize*step;m+=step) {
                    double dist = sampledist[n/step][m/step];
                    
                    if (dist>0) {
                        matrix[n/step][m/step] = affinity(dist);
                        matrix[m/step][n/step] = affinity(dist);
                    }
                }
            }
        } else {
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        matrix[n/step][m/step] = affinity(distances[d][n]);
                        matrix[m/step][n/step] = matrix[n/step][m/step];
                        
                        if (distances[d][n]>dmax) dmax = distances[d][n];
                        if (distances[d][n]>0) {
                            dmean += distances[d][n];
                            nmean++;
                        }
                    }
                }
            }
            dmean /= nmean;
            System.out.println("ngb distances (mean: "+dmean+", max:"+dmax+")");
        }

        System.out.println("..correlations");
        
        // build Laplacian
        int vol=msize;
        if (alpha>0) {
            double[] norm = new double[vol];
            for (int v1=0;v1<vol;v1++) {
                norm[v1] = 0.0;
                for (int v2=0;v2<vol;v2++) {
                    norm[v1] += matrix[v1][v2];
                }
                norm[v1] = FastMath.pow(norm[v1],-alpha);
            }
            for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                matrix[v1][v2] *= norm[v1]*norm[v2];
                matrix[v2][v1] *= norm[v2]*norm[v1];
            }
        }
            
        double[] degree = new double[vol];
        for (int v1=0;v1<vol;v1++) {
            degree[v1] = 0.0;
            for (int v2=0;v2<vol;v2++) {
                degree[v1] += matrix[v1][v2];
            }
        }
        for (int v1=0;v1<vol;v1++) {
            matrix[v1][v1] = 1.0;
        }
        for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
            matrix[v1][v2] = -matrix[v1][v2]/degree[v1];
            matrix[v2][v1] = -matrix[v2][v1]/degree[v2];
        }
        System.out.println("..Laplacian");
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
        //System.out.println("first four eigen values:");
        double[] eigval = new double[ndims+1];
        for (int s=0;s<ndims+1;s++) {
            eigval[s] = eig.getRealEigenvalues()[s];
            //System.out.print(eigval[s]+", ");
        }
        // tiled results: we should interpolate instead...
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][npt];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npt;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    sum += affinity(distances[d][n])*eig.getV().getEntry(closest[d][n]-1,dim);
                    den += affinity(distances[d][n]);
                }
                if (den>0) {
                    init[dim][n] = (float)(sum/den);
                }
            }
        }
        // refine the result with eigenGame?
        if (eigenGame && step>1) {
            System.out.println("Eigen game base volume: "+npt);
                
            // build correlation matrix
            int nmtx = 0;
            
            for (int n=0;n<npt;n++) {
                for (int m=0;m<ngbp[n].length;m++) {
                    if (ngbp[n][m]>n) nmtx++;
                }
            }
            System.out.println("non-zero components: "+nmtx);
                
            double[] mtxval = new double[nmtx];
            int[] mtxid1 = new int[nmtx];
            int[] mtxid2 = new int[nmtx];
            int[][] mtxinv = new int[nconnect][npt];
            
            int id=0;
            for (int n=0;n<npt;n++) {
                for (int m=0;m<ngbp[n].length;m++) {
                    if (ngbp[n][m]>n) {
                        int ngb = ngbp[n][m];
                        double dist = FastMath.sqrt(Numerics.square(pointList[3*n+X]-pointList[3*ngb+X])
                                                   +Numerics.square(pointList[3*n+Y]-pointList[3*ngb+Y])
                                                   +Numerics.square(pointList[3*n+Z]-pointList[3*ngb+Z]));
                        double coeff = affinity(dist);
                        mtxval[id] = coeff;
                        mtxid1[id] = n;
                        mtxid2[id] = ngb; 
                        for (int c=0;c<nconnect;c++) if (mtxinv[c][n]==0) {
                            mtxinv[c][n] = id+1;
                            c=nconnect;
                        }
                        for (int c=0;c<nconnect;c++) if (mtxinv[c][ngb]==0) {
                            mtxinv[c][ngb] = id+1;
                            c=nconnect;
                        }
                        id++;
                    }
                }
            }
            System.out.println("..correlations");
                
            // get initial vector guesses from subsampled data
            double[] norm = new double[ndims+1];
            for (int dim=0;dim<ndims+1;dim++) {
                for (int n=0;n<npt;n++) {
                    norm[dim] += init[dim][n]*init[dim][n];
                }
            }
            // rescale to ||V||=1
            for (int i=0;i<ndims+1;i++) {
                norm[i] = FastMath.sqrt(norm[i]);
                for (int vi=0;vi<npt;vi++) {
                    init[i][vi] /= norm[i];
                }
            }
                
            runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, npt, ndims, init, nconnect, alpha);
                              
        } 
        
        embeddingList = new float[npt*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]/init[0][n]);
            }
        }
        
		return;
	}
	
	public final void meshDistanceSparseEmbedding2(int depth, boolean eigenGame, boolean fullDistance, double alpha) {
	    //boolean eigenGame = true;
	    //boolean fullDistance = false;
	    //double alpha = 0.0;
	    
	    int npt=pointList.length/3;
        int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
            
	    // add a dimension for lowest eigenvector
	    //ndims = ndims+1;
	    
	    int[] samples = new int[npt];
	    for (int n=0;n<msize*step;n+=step) {
	        samples[n] = n/step+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, samples);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<npt;n++) {
            if (distances[d][n]>maxdist) maxdist = distances[d][n];
        }
	    System.out.println("fast marching distances max: "+maxdist);

	    // affinities
        //double[][] matrix = affinityMatrixFromMeshSampling(distances, closest, depth, step, msize, fullDistance);
        double[][] matrix = distanceMatrixFromMeshSampling(distances, closest, depth, step, msize, fullDistance);
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (matrix[n][m]>0) matrix[n][m] = affinity(matrix[n][m]);
        }
        
        // build Laplacian
        buildLaplacian(matrix, msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
        //System.out.println("first four eigen values:");
        double[] eigval = new double[ndims+1];
        for (int s=0;s<ndims+1;s++) {
            eigval[s] = eig.getRealEigenvalues()[s];
            //System.out.print(eigval[s]+", ");
        }
        // tiled results: we should interpolate instead...
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][npt];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npt;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    sum += affinity(distances[d][n])*eig.getV().getEntry(closest[d][n]-1,dim);
                    den += affinity(distances[d][n]);
                }
                if (den>0) {
                    init[dim][n] = (float)(sum/den);
                }
            }
        }
        // refine the result with eigenGame?
        if (eigenGame && step>1) {
            System.out.println("Eigen game base volume: "+npt);
                
            // build correlation matrix
            int nmtx = 0;
            
            for (int n=0;n<npt;n++) {
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    nmtx++;
                }
            }
            System.out.println("non-zero components: "+nmtx);
                
            double[] mtxval = new double[nmtx];
            int[] mtxid1 = new int[nmtx];
            int[] mtxid2 = new int[nmtx];
            int[][] mtxinv = new int[depth][npt];
            
            int id=0;
            for (int n=0;n<npt;n++) {
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    int ngb = closest[d][n]-1;
                        
                    double coeff = affinity(distances[d][n]);
                    mtxval[id] = coeff;
                    mtxid1[id] = n;
                    mtxid2[id] = ngb; 
                    mtxinv[d][n] = id+1;
                    mtxinv[d][ngb] = id+1;
                    id++;
                }
            }
            System.out.println("..correlations");
                
            // get initial vector guesses from subsampled data
            double[] norm = new double[ndims+1];
            for (int dim=0;dim<ndims+1;dim++) {
                for (int n=0;n<npt;n++) {
                    norm[dim] += init[dim][n]*init[dim][n];
                }
            }
            // rescale to ||V||=1
            for (int i=0;i<ndims+1;i++) {
                norm[i] = FastMath.sqrt(norm[i]);
                for (int vi=0;vi<npt;vi++) {
                    init[i][vi] /= norm[i];
                }
            }
                
            runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, npt, ndims, init, depth, alpha);
                              
        } 
        
        embeddingList = new float[npt*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]/init[0][n]);
            }
        }
        
		return;
	}
	
	public final void meshDistanceJointSparseEmbedding(int depth, boolean eigenGame, boolean fullDistance, double alpha) {
	    //boolean eigenGame = true;
	    //boolean fullDistance = false;
	    //double alpha = 0.0;
	    
	    int npt=pointList.length/3;
        int step = Numerics.floor(npt/msize);
        
	    int nrf=pointListRef.length/3;
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step sizes: "+step+", "+stpf);
            
	    // add a dimension for lowest eigenvector
	    //ndims = ndims+1;
	    
	    int[] samples = new int[npt];
	    for (int n=0;n<msize*step;n+=step) {
	        samples[n] = n/step+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, samples);
        
	    int[] samplesRef = new int[nrf];
	    for (int n=0;n<msize*stpf;n+=stpf) {
	        samplesRef[n] = n/stpf+1;
	    }
	    float[][] distancesRef = new float[depth][nrf];
        int[][] closestRef = new int[depth][nrf];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distancesRef, closestRef, pointListRef, triangleListRef, samplesRef);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<npt;n++) {
            if (distances[d][n]>maxdist) maxdist = distances[d][n];
        }
	    float maxdistRef = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<nrf;n++) {
            if (distancesRef[d][n]>maxdistRef) maxdistRef = distancesRef[d][n];
        }
	    System.out.println("fast marching distances max: "+maxdist+", "+maxdistRef);

	    // affinities
        double[][] matrix = distanceMatrixFromMeshSampling(distances, closest, depth, step, msize, fullDistance);
        double[][] matrixRef = distanceMatrixFromMeshSampling(distancesRef, closestRef, depth, stpf, msize, fullDistance);
        
        // linking functions
        double[][] linker = new double[msize][msize];
        for (int n=0;n<msize;n++) for (int m=n;m<msize;m++) {
            // linking distance: average of geodesic and point distances
            double distN = FastMath.sqrt(Numerics.square(pointList[3*n+X]-pointListRef[3*n+X])
                                        +Numerics.square(pointList[3*n+Y]-pointListRef[3*n+Y])
                                        +Numerics.square(pointList[3*n+Z]-pointListRef[3*n+Z]));

            double distM = FastMath.sqrt(Numerics.square(pointList[3*m+X]-pointListRef[3*m+X])
                                        +Numerics.square(pointList[3*m+Y]-pointListRef[3*m+Y])
                                        +Numerics.square(pointList[3*m+Z]-pointListRef[3*m+Z]));

            //linker[n][m] = 0.5*(matrix[n][m]+matrixRef[n][m]+distN+distM);
            linker[n][m] = 0.5*(matrix[n][m]+matrixRef[n][m]);
            linker[m][n] = linker[n][m];
        }
        
        double[][] joint = new double[2*msize][2*msize];
        for (int n=0;n<msize;n++) for (int m=n;m<msize;m++) {
            joint[n][m] = affinity(matrix[n][m]);
            joint[n+msize][m] = linking(linker[n][m]);
            joint[n][m+msize] = joint[n+msize][m];
            joint[n+msize][m+msize] = affinity(matrixRef[n][m]);
        }
        
        // build Laplacian
        buildLaplacian(joint, 2*msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(joint);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
        //System.out.println("first four eigen values:");
        double[] eigval = new double[ndims+1];
        int[] eignum = new int[ndims+1];
        eigval[0] = 1e16;
        for (int n=0;n<2*msize;n++) {
            if (eig.getRealEigenvalues()[n]<eigval[0]) {
                eigval[0] = eig.getRealEigenvalues()[n];
                eignum[0] = n;
            }
        }
        for (int s=1;s<ndims+1;s++) {
            eigval[s] = 1e16;
            for (int n=0;n<2*msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[s] && eig.getRealEigenvalues()[n]>eigval[s-1]) {
                    eigval[s] = eig.getRealEigenvalues()[n];
                    eignum[s] = n;
                }
            }
        }
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][npt+nrf];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npt;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    sum += affinity(distances[d][n])*eig.getV().getEntry(closest[d][n]-1,eignum[dim]);
                    den += affinity(distances[d][n]);
                }
                if (den>0) {
                    init[dim][n] = (float)(sum/den);
                }
            }
            for (int n=0;n<nrf;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    sum += affinity(distancesRef[d][n])*eig.getV().getEntry(msize+closestRef[d][n]-1,eignum[dim]);
                    den += affinity(distancesRef[d][n]);
                }
                if (den>0) {
                    init[dim][npt+n] = (float)(sum/den);
                }
            }
        }
        
        // refine the result with eigenGame?
        if (eigenGame && step>1) {
            System.out.println("Eigen game base volume: "+npt+nrf);
                
            // build correlation matrix
            int nmtx = 0;
            
            for (int n=0;n<npt;n++) {
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    nmtx++;
                }
            }
            for (int n=0;n<nrf;n++) {
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    nmtx++;
                }
            }
            System.out.println("non-zero components: "+nmtx);
                
            double[] mtxval = new double[nmtx];
            int[] mtxid1 = new int[nmtx];
            int[] mtxid2 = new int[nmtx];
            int[][] mtxinv = new int[depth][npt+nrf];
            
            int id=0;
            for (int n=0;n<npt;n++) {
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    int ngb = closest[d][n]-1;
                        
                    double coeff = affinity(distances[d][n]);
                    mtxval[id] = coeff;
                    mtxid1[id] = n;
                    mtxid2[id] = ngb; 
                    mtxinv[d][n] = id+1;
                    mtxinv[d][ngb] = id+1;
                    id++;
                }
            }
            for (int n=0;n<nrf;n++) {
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    int ngb = closestRef[d][n]-1;
                        
                    double coeff = affinity(distancesRef[d][n]);
                    mtxval[id] = coeff;
                    mtxid1[id] = npt+n;
                    mtxid2[id] = npt+ngb; 
                    mtxinv[d][npt+n] = id+1;
                    mtxinv[d][npt+ngb] = id+1;
                    id++;
                }
            }
            System.out.println("..correlations");
                
            // get initial vector guesses from subsampled data
            double[] norm = new double[ndims+1];
            for (int dim=0;dim<ndims+1;dim++) {
                for (int n=0;n<npt;n++) {
                    norm[dim] += init[dim][n]*init[dim][n];
                }
            }
            // rescale to ||V||=1
            for (int i=0;i<ndims+1;i++) {
                norm[i] = FastMath.sqrt(norm[i]);
                for (int vi=0;vi<npt;vi++) {
                    init[i][vi] /= norm[i];
                }
            }
                
            runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, npt, ndims, init, depth, alpha);
                              
        } 
        
        embeddingList = new float[npt*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<npt;n++) {
                //embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]/init[0][n]);
                embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]);
            }
        }
        
        embeddingListRef = new float[nrf*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<nrf;n++) {
                //embeddingListRef[n+(dim-1)*nrf] = (float)(init[dim][npt+n]/init[0][npt+n]);
                embeddingListRef[n+(dim-1)*nrf] = (float)(init[dim][npt+n]);
            }
        }
        
		return;
	}
	
	public final void meshDistanceReferenceSparseEmbedding(int depth, boolean eigenGame, boolean fullDistance, double alpha) {
	    //boolean eigenGame = true;
	    //boolean fullDistance = false;
	    //double alpha = 0.0;
	    
	    int nrf=pointListRef.length/3;
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step size: "+stpf);
            
	    // add a dimension for lowest eigenvector
	    //ndims = ndims+1;
	    
	    int[] samplesRef = new int[nrf];
	    for (int n=0;n<msize*stpf;n+=stpf) {
	        samplesRef[n] = n/stpf+1;
	    }
	    float[][] distancesRef = new float[depth][nrf];
        int[][] closestRef = new int[depth][nrf];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distancesRef, closestRef, pointListRef, triangleListRef, samplesRef);
        
	    float maxdistRef = 0.0f;
        for (int d=0;d<depth;d++) for (int n=0;n<nrf;n++) {
            if (distancesRef[d][n]>maxdistRef) maxdistRef = distancesRef[d][n];
        }
	    System.out.println("fast marching distances max: "+maxdistRef);

	    // affinities
        double[][] matrixRef = distanceMatrixFromMeshSampling(distancesRef, closestRef, depth, stpf, msize, fullDistance);
        
        // linking function is affinity
        double[][] joint = new double[2*msize][2*msize];
        for (int n=0;n<msize;n++) for (int m=n;m<msize;m++) {
            joint[n][m] = affinity(matrixRef[n][m]);
            joint[n+msize][m] = linking(matrixRef[n][m]);
            joint[n][m+msize] = joint[n+msize][m];
            joint[n+msize][m+msize] = joint[n][m];
        }
        
        // build Laplacian
        buildLaplacian(joint, 2*msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(joint);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
        //System.out.println("first four eigen values:");
        double[] eigval = new double[ndims+1];
        for (int s=0;s<ndims+1;s++) {
            eigval[s] = eig.getRealEigenvalues()[s];
            //System.out.print(eigval[s]+", ");
        }
        // from mean coord to neighbors
        double[][] initRef = new double[ndims+1][nrf];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<nrf;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    sum += affinity(distancesRef[d][n])*eig.getV().getEntry(msize+closestRef[d][n]-1,dim);
                    den += affinity(distancesRef[d][n]);
                }
                if (den>0) {
                    initRef[dim][n] = (float)(sum/den);
                }
            }
        }
        
        // refine the result with eigenGame?
        if (eigenGame && stpf>1) {
            System.out.println("Eigen game base volume: "+nrf);
                
            // build correlation matrix
            int nmtx = 0;
            
            for (int n=0;n<nrf;n++) {
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    nmtx++;
                }
            }
            System.out.println("non-zero components: "+nmtx);
                
            double[] mtxval = new double[nmtx];
            int[] mtxid1 = new int[nmtx];
            int[] mtxid2 = new int[nmtx];
            int[][] mtxinv = new int[depth][nrf];
            
            int id=0;
            for (int n=0;n<nrf;n++) {
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    int ngb = closestRef[d][n]-1;
                        
                    double coeff = affinity(distancesRef[d][n]);
                    mtxval[id] = coeff;
                    mtxid1[id] = n;
                    mtxid2[id] = ngb; 
                    mtxinv[d][n] = id+1;
                    mtxinv[d][ngb] = id+1;
                    id++;
                }
            }
            System.out.println("..correlations");
                
            // get initial vector guesses from subsampled data
            double[] norm = new double[ndims+1];
            for (int dim=0;dim<ndims+1;dim++) {
                for (int n=0;n<nrf;n++) {
                    norm[dim] += initRef[dim][n]*initRef[dim][n];
                }
            }
            // rescale to ||V||=1
            for (int i=0;i<ndims+1;i++) {
                norm[i] = FastMath.sqrt(norm[i]);
                for (int vi=0;vi<nrf;vi++) {
                    initRef[i][vi] /= norm[i];
                }
            }
                
            runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, nrf, ndims, initRef, depth, alpha);
                              
        } 
        
        embeddingListRef = new float[nrf*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<nrf;n++) {
                embeddingListRef[n+(dim-1)*nrf] = (float)(initRef[dim][n]/initRef[0][n]);
            }
        }
        
		return;
	}
	
	private final double[][] affinityMatrixFromMeshSampling(float[][] distances, int[][] closest, int depth, int step, int msize, boolean fullDistance) {
	    
        // precompute surface-based distances
	    double[][] matrix = new double[msize][msize];
	    
	    if (fullDistance) {
            // build a complete sample distance map? should be doable, roughly O(msize^2)
            // very slow for large meshes, though, and maybe not needed anyway given the eigengame step
            float[][] sampledist = new float[msize][msize];
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        sampledist[n/step][m/step] = distances[d][n];
                        sampledist[m/step][n/step] = distances[d][n];
                    }
                }
            }
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                if (sampledist[n/step][m/step]>0) {
                    dmean += sampledist[n/step][m/step];
                    nmean++;
                }
            }
            dmean /= nmean;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
    
            // set to false to skip the propagation
            int missing=1;
            int prev = -1;
            int nmiss=0;
            while (missing>0 && missing!=prev) {
                prev = missing;
                missing=0;
                for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                    if (sampledist[n/step][m/step]>0) {
                        for (int d=0;d<depth;d++) {
                            int jm = (closest[d][n]-1)*step;
                            if (jm>=0) {
                                if (sampledist[jm/step][m/step]==0) sampledist[jm/step][m/step] = distances[d][n]+sampledist[n/step][m/step];
                                else sampledist[jm/step][m/step] = Numerics.min(sampledist[jm/step][m/step],distances[d][n]+sampledist[n/step][m/step]);
                                sampledist[m/step][jm/step] = sampledist[jm/step][m/step];
                            }
                            int jn = (closest[d][m]-1)*step;
                            if (jn>=0) {
                                if (sampledist[jn/step][n/step]==0) sampledist[jn/step][n/step] = distances[d][m]+sampledist[n/step][m/step];
                                else sampledist[jn/step][n/step] = Numerics.min(sampledist[jn/step][n/step],distances[d][m]+sampledist[n/step][m/step]);
                                sampledist[n/step][jn/step] = sampledist[jn/step][n/step];
                            }
                        }
                    } else {
                        missing++;
                    }
                }
                nmiss++;
            }
            System.out.println("approximate distance propagation: "+nmiss);
            dmax=0.0f; 
            dmean=0.0f;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                dmean += sampledist[n/step][m/step];
            }
            dmean /= msize*msize;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
            
            // reset diagonal to zero to have correct distance when closest
            for (int n=0;n<msize*step;n+=step) {
                sampledist[n/step][n/step] = 0.0f;
            }
            
            if (scale<0) scale = dmean;
            
            for (int n=0;n<msize*step;n+=step) {
                for (int m=n+step;m<msize*step;m+=step) {
                    double dist = sampledist[n/step][m/step];
                    
                    if (dist>0) {
                        matrix[n/step][m/step] = affinity(dist);
                        matrix[m/step][n/step] = affinity(dist);
                    }
                }
            }
        } else {
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        matrix[n/step][m/step] = affinity(distances[d][n]);
                        matrix[m/step][n/step] = matrix[n/step][m/step];
                        
                        if (distances[d][n]>dmax) dmax = distances[d][n];
                        if (distances[d][n]>0) {
                            dmean += distances[d][n];
                            nmean++;
                        }
                    }
                }
            }
            dmean /= nmean;
            System.out.println("ngb distances (mean: "+dmean+", max:"+dmax+")");
        }
        return matrix;
    }
    
	private final double[][] distanceMatrixFromMeshSampling(float[][] distances, int[][] closest, int depth, int step, int msize, boolean fullDistance) {
	    
        // precompute surface-based distances
	    double[][] matrix = new double[msize][msize];
	    
	    if (fullDistance) {
            // build a complete sample distance map? should be doable, roughly O(msize^2)
            // very slow for large meshes, though, and maybe not needed anyway given the eigengame step
            float[][] sampledist = new float[msize][msize];
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        sampledist[n/step][m/step] = distances[d][n];
                        sampledist[m/step][n/step] = distances[d][n];
                    }
                }
            }
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                if (sampledist[n/step][m/step]>0) {
                    dmean += sampledist[n/step][m/step];
                    nmean++;
                }
            }
            dmean /= nmean;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
    
            // set to false to skip the propagation
            int missing=1;
            int prev = -1;
            int nmiss=0;
            while (missing>0 && missing!=prev) {
                prev = missing;
                missing=0;
                for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                    if (sampledist[n/step][m/step]>0) {
                        for (int d=0;d<depth;d++) {
                            int jm = (closest[d][n]-1)*step;
                            if (jm>=0) {
                                if (sampledist[jm/step][m/step]==0) sampledist[jm/step][m/step] = distances[d][n]+sampledist[n/step][m/step];
                                else sampledist[jm/step][m/step] = Numerics.min(sampledist[jm/step][m/step],distances[d][n]+sampledist[n/step][m/step]);
                                sampledist[m/step][jm/step] = sampledist[jm/step][m/step];
                            }
                            int jn = (closest[d][m]-1)*step;
                            if (jn>=0) {
                                if (sampledist[jn/step][n/step]==0) sampledist[jn/step][n/step] = distances[d][m]+sampledist[n/step][m/step];
                                else sampledist[jn/step][n/step] = Numerics.min(sampledist[jn/step][n/step],distances[d][m]+sampledist[n/step][m/step]);
                                sampledist[n/step][jn/step] = sampledist[jn/step][n/step];
                            }
                        }
                    } else {
                        missing++;
                    }
                }
                nmiss++;
            }
            System.out.println("approximate distance propagation: "+nmiss);
            dmax=0.0f; 
            dmean=0.0f;
            for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
                if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
                dmean += sampledist[n/step][m/step];
            }
            dmean /= msize*msize;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
            
            // reset diagonal to zero to have correct distance when closest
            for (int n=0;n<msize*step;n+=step) {
                sampledist[n/step][n/step] = 0.0f;
            }
            
            if (scale<0) scale = dmean;
            
            for (int n=0;n<msize*step;n+=step) {
                for (int m=n+step;m<msize*step;m+=step) {
                    double dist = sampledist[n/step][m/step];
                    
                    if (dist>0) {
                        matrix[n/step][m/step] = dist;
                        matrix[m/step][n/step] = dist;
                    }
                }
            }
        } else {
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize*step;n+=step) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][n]-1)*step;
                    if (m>=0) {
                        matrix[n/step][m/step] = distances[d][n];
                        matrix[m/step][n/step] = matrix[n/step][m/step];
                        
                        if (distances[d][n]>dmax) dmax = distances[d][n];
                        if (distances[d][n]>0) {
                            dmean += distances[d][n];
                            nmean++;
                        }
                    }
                }
            }
            dmean /= nmean;
            System.out.println("ngb distances (mean: "+dmean+", max:"+dmax+")");
        }
        return matrix;
    }
    
    private final void buildLaplacian(double[][] matrix, int vol, double alpha) {
        if (alpha>0) {
            double[] norm = new double[vol];
            for (int v1=0;v1<vol;v1++) {
                norm[v1] = 0.0;
                for (int v2=0;v2<vol;v2++) {
                    norm[v1] += matrix[v1][v2];
                }
                norm[v1] = FastMath.pow(norm[v1],-alpha);
            }
            for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                matrix[v1][v2] *= norm[v1]*norm[v2];
                matrix[v2][v1] *= norm[v2]*norm[v1];
            }
        }
            
        double[] degree = new double[vol];
        for (int v1=0;v1<vol;v1++) {
            degree[v1] = 0.0;
            for (int v2=0;v2<vol;v2++) {
                degree[v1] += matrix[v1][v2];
            }
        }
        for (int v1=0;v1<vol;v1++) {
            matrix[v1][v1] = 1.0;
        }
        for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
            matrix[v1][v2] = -matrix[v1][v2]/degree[v1];
            matrix[v2][v1] = -matrix[v2][v1]/degree[v2];
        }
        System.out.println("..Laplacian");   
    }

	private final void runSparseLaplacianEigenGame(double[] mtval, int[] mtid1, int[] mtid2, int[][] mtinv, int nn0, int nm, int nv, double[][] init, int nconnect, double alpha) {
        //double step = 1e-2;     // step size
	    //double step = 1e-1;     // step size
	    double step = 0.05;     // step size
	    //double error = 1e-2;    // error tolerance: makes a big difference in max steps...
	    double error = 0.05;    // error tolerance
	    int iter;
        double[][] Mv = new double[nv][nm];
        double[] vMv = new double[nv];
        
        double[][] vect = init;
        
        // here assume the matrix is the upper diagonal of correlation matrix
        
        // correction for different norms
        if (alpha>0) {
            double[] norm = new double[nm];
            for (int n=0;n<nn0;n++) {
                norm[mtid1[n]] += mtval[n];
                norm[mtid2[n]] += mtval[n];
            }
            for (int n=0;n<nm;n++) {
                norm[n] = FastMath.pow(norm[n],-alpha);
            }
            for (int n=0;n<nn0;n++) {
                mtval[n] *= norm[mtid1[n]]*norm[mtid2[n]];
            }
        }           

        // build degree first
        double[] deg = new double[nm];
        // M_ii = 0
        for (int n=0;n<nm;n++) {
            deg[n] = 0.0;
        }
        // M_ij and M_ji
        for (int n=0;n<nn0;n++) {
            deg[mtid1[n]] += mtval[n];
            deg[mtid2[n]] += mtval[n];
        }
        
        for (int vi=0;vi<nv;vi++) {
            System.out.println("..eigenvector "+(vi+1));
        
            // compute new vectors based on 
            for (int n=0;n<nm;n++) {
                /* generic formula
                Mv[vi][n] = 0.0;
                for (int m=0;m<nm;m++)
                    Mv[vi][n] += matrix[n][m]*vect[vi][m];
                    */
                // diagonal term is 2-1, as lambda_0<=2 (graph Laplacian property)
                Mv[vi][n] = vect[vi][n];
                //Mv[vi][n] = vect[vi][n]/mtw[n];
                // off-diagonals
                for (int c=0;c<nconnect;c++) if (mtinv[c][n]>0) {
                    if (mtid1[mtinv[c][n]-1]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid2[mtinv[c][n]-1]];
                        //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid2[mtinv[c][n]-1]];
                    } else if (mtid2[mtinv[c][n]-1]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid1[mtinv[c][n]-1]];
                        //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid1[mtinv[c][n]-1]];
                    }  
                }
                /*
                for (int m=0;m<nn0;m++) {
                    if (mtid1[m]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[m]/deg[n]*vect[vi][mtid2[m]];
                    } else if (mtid2[m]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[m]/deg[n]*vect[vi][mtid1[m]];
                    }
                }*/
            }
            
            // calculate required number of iterations
            double norm = 0.0;
            for (int n=0;n<nm;n++) norm += Mv[vi][n]*Mv[vi][n];
            System.out.println("norm: "+norm);
            
            double Ti = 5.0/4.0/Numerics.min(norm/4.0, error*error);
            System.out.println("-> "+Ti+" iterations");
            
            // pre-compute previous quantities?
            
            // main loop
            double[] grad = new double[nm];
//            for (int t=0;t<Ti;t++) {
            int t=0;
            while (t<Ti && Numerics.abs(norm/4.0-1.0)>error*error) {
                t++;
                //System.out.print(".");
                // pre-compute product
                double[] viMvj = new double[nv];
                for (int vj=0;vj<vi;vj++) {
                    viMvj[vj] = 0.0;
                    for (int m=0;m<nm;m++) viMvj[vj] += Mv[vj][m]*vect[vi][m];
                }
                // gradient computation
                for (int n=0;n<nm;n++) {
                    grad[n] = 2.0*Mv[vi][n];
                    for (int vj=0;vj<vi;vj++) {
                        //double prod = 0.0;
                        //for (int m=0;m<nm;m++) prod += Mv[vj][m]*vect[vi][m];
                        grad[n] -= 2.0*viMvj[vj]/vMv[vj]*Mv[vj][n];
                    }
                }
                // Riemannian projection
                double gradR = 0.0;
                for (int n=0;n<nm;n++)
                    gradR += grad[n]*vect[vi][n];
                
                // update
                norm = 0.0;
                for (int n=0;n<nm;n++) {
                    vect[vi][n] += step*(grad[n] - gradR*vect[vi][n]);
                    norm += vect[vi][n]*vect[vi][n];
                }
                norm = FastMath.sqrt(norm);
                
                // renormalize 
                for (int n=0;n<nm;n++) {
                    vect[vi][n] /= norm;
                }
                
                // recompute Mvi
                for (int n=0;n<nm;n++) {
                    /* replace by compressed matrix
                    Mv[vi][n] = 0.0;
                    for (int m=0;m<nm;m++)
                        Mv[vi][n] += matrix[n][m]*vect[vi][m];
                        */
                    // diagonal term is 2-1
                    Mv[vi][n] = vect[vi][n];
                    //Mv[vi][n] = vect[vi][n]/mtw[n];
                    // off-diagonals
                    for (int c=0;c<nconnect;c++) if (mtinv[c][n]>0) {
                        if (mtid1[mtinv[c][n]-1]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid2[mtinv[c][n]-1]];
                            //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid2[mtinv[c][n]-1]];
                        } else if (mtid2[mtinv[c][n]-1]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid1[mtinv[c][n]-1]];
                            //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid1[mtinv[c][n]-1]];
                        }  
                    }
                    /*
                    // off-diagonals
                    for (int m=0;m<nn0;m++) {
                        if (mtid1[m]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[m]/deg[mtid1[m]]*vect[vi][mtid2[m]];
                        } else if (mtid2[m]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[m]/deg[mtid2[m]]*vect[vi][mtid1[m]];
                        }
                    }*/
                }
    
                // recompute norm to stop earlier if possible?
                norm = 0.0;
                for (int n=0;n<nm;n++) norm += Mv[vi][n]*Mv[vi][n];
            }
            System.out.println(" ("+t+" needed, norm: "+norm+")");
            //System.out.println("norm: "+norm);
            
            // post-process: compute summary quantities for next eigenvector
            vMv[vi] = 0.0;
            for (int n=0;n<nm;n++) vMv[vi] += vect[vi][n]*Mv[vi][n];
        }
        
        // check the result
        System.out.println("final vector orthogonality");
        for (int v1=0;v1<nv-1;v1++) for (int v2=v1+1;v2<nv;v2++) {
            double prod=0.0;
            for (int m=0;m<nm;m++) prod += vect[v1][m]*vect[v2][m];
            System.out.println("v"+v1+" * v"+v2+" = "+prod);
        }
        System.out.println("final vector eigenscore");
        for (int v1=0;v1<nv;v1++) {
            double normvect=0.0;
            double normMv=0.0;
            double prod=0.0;
            for (int m=0;m<nm;m++) {
                normvect += vect[v1][m]*vect[v1][m];
                normMv += Mv[v1][m]*Mv[v1][m];
                prod += vect[v1][m]*Mv[v1][m];
            }
            System.out.println("v"+v1+" . Mv"+v1+" = "+prod/FastMath.sqrt(normvect*normMv)+" (lambda = "+normMv/normvect+")");
        }
    }

    private final void embeddingReferenceRotation(double[][] ref0, double[][] ref1, double[][] sub1) {
        
        int ndims = sub1.length;
        int npt = sub1[0].length;
        int nrf = ref1[0].length;
        
        // build a rotation matrix for reference 1 to 0
        double[] norm0 = new double[ndims];
	    double[] norm1 = new double[ndims];
	    for (int n=0;n<ndims;n++) {
	        norm0[n] = 0.0;
	        norm1[n] = 0.0;
	        for (int i=0;i<nrf;i++) {
	            norm0[n] += ref0[n][i]*ref0[n][i];
	            norm1[n] += ref1[n][i]*ref1[n][i];
	        }
	        norm0[n] = FastMath.sqrt(norm0[n]);
	        norm1[n] = FastMath.sqrt(norm1[n]);
	    }
	    double[][] rot = new double[ndims][ndims];
	    for (int m=0;m<ndims;m++) for (int n=0;n<ndims;n++) {
	        rot[m][n] = 0.0;
	        for (int i=0;i<nrf;i++) {
	            rot[m][n] += ref1[m][i]/norm1[m]*ref0[n][i]/norm0[n];
	        }
	    }
	    System.out.println("rotation matrix");
	    for (int m=0;m<ndims;m++) {
	        System.out.print("[ ");
	        for (int n=0;n<ndims;n++) {
	            System.out.print(rot[m][n]+" ");
	        }
	        System.out.println("]");
	    }
	    double[][] rotated = new double[ndims][npt];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int j=0;j<npt;j++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += sub1[m][j]*rot[m][n];
	            }
	            rotated[n][j] = val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int j=0;j<npt;j++) {
	            rotated[n][j] /= norm;
	        }
	    }
	    sub1 = rotated;
    }

    private final void embeddingReferenceRotation(float[] ref0, float[] ref1, float[] sub1, int nrf, int npt, int ndims) {
        
        // build a rotation matrix for reference 1 to 0
        double[] norm0 = new double[ndims];
	    double[] norm1 = new double[ndims];
	    for (int n=0;n<ndims;n++) {
	        norm0[n] = 0.0;
	        norm1[n] = 0.0;
	        for (int i=0;i<nrf;i++) {
	            norm0[n] += ref0[i+n*nrf]*ref0[i+n*nrf];
	            norm1[n] += ref1[i+n*nrf]*ref1[i+n*nrf];
	        }
	        norm0[n] = FastMath.sqrt(norm0[n]);
	        norm1[n] = FastMath.sqrt(norm1[n]);
	    }
	    double[][] rot = new double[ndims][ndims];
	    for (int m=0;m<ndims;m++) for (int n=0;n<ndims;n++) {
	        rot[m][n] = 0.0;
	        for (int i=0;i<nrf;i++) {
	            rot[m][n] += ref1[i+m*nrf]/norm1[m]*ref0[i+n*nrf]/ref0[n];
	        }
	    }
	    System.out.println("rotation matrix");
	    for (int m=0;m<ndims;m++) {
	        System.out.print("[ ");
	        for (int n=0;n<ndims;n++) {
	            System.out.print(rot[m][n]+" ");
	        }
	        System.out.println("]");
	    }
	    float[] rotated = new float[ndims*npt];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int j=0;j<npt;j++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += sub1[j+m*npt]*rot[m][n];
	            }
	            rotated[j+n*npt] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int j=0;j<npt;j++) {
	            rotated[j+n*npt] /= (float)norm;
	        }
	    }
	    sub1 = rotated;
    }
    
    public void rotatedJointSpatialEmbedding(int depth, boolean eigenGame, boolean fullDistance, double alpha) {

	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    meshDistanceReferenceSparseEmbedding(depth, eigenGame, fullDistance, alpha);
	    float[] refEmbedding = new float[embeddingListRef.length];
	    for (int n=0;n<embeddingListRef.length;n++) {
	        refEmbedding[n] = embeddingListRef[n];
	    }
	    
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    meshDistanceJointSparseEmbedding(depth, eigenGame, fullDistance, alpha);
	    	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    int npt=pointList.length/3;
        int nrf=pointListRef.length/3;
        embeddingReferenceRotation(refEmbedding, embeddingListRef, embeddingList, nrf, npt, ndims);
	}

}