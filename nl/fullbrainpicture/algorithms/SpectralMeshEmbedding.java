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
					
	// create outputs
	public final float[] 	getEmbeddingValues() { return embeddingList; }
	public final float[] 	getReferenceEmbeddingValues() { return embeddingListRef; }
	
	public void pointDistanceEmbedding(){
	    
	    // data size
	    int npt = pointList.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
        
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
	        // self affinitiy should be 1?
            Azero[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {	            
	            // for now: approximate geodesic distance with Euclidean distance
	            // note that it is not an issue for data-based distance methods
	            double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	                         
	            Azero[n/step][m/step] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[m/step][n/step] = Azero[n/step][m/step];
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
	    /* not needed
	    for (int n=0;n<msize*step;n+=step) {
	        a0r[n/step] = 0.0;
	        b0r[n/step] = 0.0;
	    }*/
	    for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {	    
	            /*
	            double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	                         
	            a0r[n/step] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	            */
	            a0r[n/step] += Azero[n/step][m/step];
	        }
	        for (int m=0;m<npt;m++) if (m%step!=0 || m>=msize*step) {
	            double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	            
	            b0r[n/step] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
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
	    /*
	    for (int n=0;n<npt;n++) {
	        degree[n] = 0.0;
	    }*/
	    for (int n=0;n<msize*step;n+=step) {
	        degree[n] = a0r[n/step]+b0r[n/step];
	    }
	    for (int n=0;n<npt;n++) if (n%step!=0 || n>=msize*step) {
	        for (int m=0;m<msize*step;m+=step) {	
                double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
                             +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
                             +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
            
                degree[n] += (1.0 + ainvb0r[m/step])*1.0/(1.0+FastMath.sqrt(dist)/scale);
            }
	    }

        System.out.println("build first approximation");
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
            Acore[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {	            
	            // for now: approximate geodesic distance with Euclidean distance
	            // note that it is not an issue for data-based distance methods
	            /*
	            double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	                         
	            Acore[n/step][m/step] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n];
                Acore[m/step][n/step] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[m];
                */
                Acore[n/step][m/step] = -Azero[n/step][m/step]/degree[n];
                Acore[m/step][n/step] = -Azero[m/step][n/step]/degree[m];
            }
        }
	    // First decomposition: A
        mtx = new Array2DRowRealMatrix(Acore);
        eig = new EigenDecomposition(mtx);
        
        // sort eigenvalues: not needed, already done :)
        /*
        embeddingList = new float[pointList.length/3*ndims];
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+eig.getRealEigenvalue(d));
            for (int n=0;n<msize*step;n+=step) {
                embeddingList[n+d*pointList.length/3] = (float)eig.getEigenvector(d).getEntry(n/step);
            }
        }
        */
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
        
        for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {
	            BBt[n/step][m/step] = 0.0;
	            for (int j=0;j<npt;j++) if (j%step!=0 || j>=msize*step) {
	                double distN = Numerics.square(pointList[3*n+X]-pointList[3*j+X])
	                              +Numerics.square(pointList[3*n+Y]-pointList[3*j+Y])
	                              +Numerics.square(pointList[3*n+Z]-pointList[3*j+Z]);
	                              
	                double distM = Numerics.square(pointList[3*m+X]-pointList[3*j+X])
	                              +Numerics.square(pointList[3*m+Y]-pointList[3*j+Y])
	                              +Numerics.square(pointList[3*m+Z]-pointList[3*j+Z]);
	                              
	                BBt[n/step][m/step] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                      *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
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
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npt;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    
                    double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
                                 +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
                                 +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
                                 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/step][d];
                }
                embeddingList[n+d*npt] = (float)embed;
            }
        }
        
		return;
	}

	public void pointDistanceJointEmbedding(){
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npt = pointList.length/3;
	    int nrf = pointListRef.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npt/msize+nrf/msize);
	    System.out.println("step size: "+step);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    double[] degree = new double[npt+nrf];
	    for (int n=0;n<npt;n++) {
	        degree[n] = 0.0;
	        for (int m=0;m<npt;m++) {
	            double dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<nrf;m++) {
	            double dist = Numerics.square(pointList[3*n+X]-pointListRef[3*m+X])
	                         +Numerics.square(pointList[3*n+Y]-pointListRef[3*m+Y])
	                         +Numerics.square(pointList[3*n+Z]-pointListRef[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
	    for (int n=0;n<nrf;n++) {
	        degree[n+pointList.length/3] = 0.0;
	        for (int m=0;m<pointList.length/3;m++) {
	            double dist = Numerics.square(pointListRef[3*n+X]-pointList[3*m+X])
	                         +Numerics.square(pointListRef[3*n+Y]-pointList[3*m+Y])
	                         +Numerics.square(pointListRef[3*n+Z]-pointList[3*m+Z]);
	            
	            degree[n+npt] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<nrf;m++) {
	            double dist = Numerics.square(pointListRef[3*n+X]-pointListRef[3*m+X])
	                         +Numerics.square(pointListRef[3*n+Y]-pointListRef[3*m+Y])
	                         +Numerics.square(pointListRef[3*n+Z]-pointListRef[3*m+Z]);
	            
	            degree[n+npt] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
            Acore[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {
	            double dist;
	            if (n<npt && m<npt) {
	                dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
	                      +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
	                      +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
	            } else if (n>=npt && m<npt) {
	                dist = Numerics.square(pointListRef[3*(n-npt)+X]-pointList[3*m+X])
	                      +Numerics.square(pointListRef[3*(n-npt)+Y]-pointList[3*m+Y])
	                      +Numerics.square(pointListRef[3*(n-npt)+Z]-pointList[3*m+Z]);
	            } else if (n<npt && m>=npt) {
	                dist = Numerics.square(pointList[3*n+X]-pointListRef[3*(m-npt)+X])
	                      +Numerics.square(pointList[3*n+Y]-pointListRef[3*(m-npt)+Y])
	                      +Numerics.square(pointList[3*n+Z]-pointListRef[3*(m-npt)+Z]);
	            } else {
	                dist = Numerics.square(pointListRef[3*(n-npt)+X]-pointList[3*(m-npt)+X])
	                      +Numerics.square(pointListRef[3*(n-npt)+Y]-pointList[3*(m-npt)+Y])
	                      +Numerics.square(pointListRef[3*(n-npt)+Z]-pointList[3*(m-npt)+Z]);
	            }        
	            Acore[n/step][m/step] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n];
                Acore[m/step][n/step] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[m];
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        // sort eigenvalues: not needed, already done :)
        /*
        embeddingList = new float[pointList.length/3*ndims];
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+eig.getRealEigenvalue(d));
            for (int n=0;n<msize*step;n+=step) {
                embeddingList[n+d*pointList.length/3] = (float)eig.getEigenvector(d).getEntry(n/step);
            }
        }
        */
        
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
        
        // banded matrix is too big, we only compute BB^T
        double[][] BBt = new double[msize][msize];
        
        for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {
	            BBt[n/step][m/step] = 0.0;
	            for (int j=0;j<npt;j++) if (j%step!=0) {
	                double distN;
	                if (n<npt) {
	                    distN = Numerics.square(pointList[3*n+X]-pointList[3*j+X])
	                           +Numerics.square(pointList[3*n+Y]-pointList[3*j+Y])
	                           +Numerics.square(pointList[3*n+Z]-pointList[3*j+Z]);
	                } else {
	                    distN = Numerics.square(pointListRef[3*(n-npt)+X]-pointList[3*j+X])
	                           +Numerics.square(pointListRef[3*(n-npt)+Y]-pointList[3*j+Y])
	                           +Numerics.square(pointListRef[3*(n-npt)+Z]-pointList[3*j+Z]);
	                }
	                double distM;
	                if (m<npt) {
	                    distM = Numerics.square(pointList[3*m+X]-pointList[3*j+X])
	                           +Numerics.square(pointList[3*m+Y]-pointList[3*j+Y])
	                           +Numerics.square(pointList[3*m+Z]-pointList[3*j+Z]);
	                } else {   
	                    distM = Numerics.square(pointListRef[3*(m-npt)+X]-pointList[3*j+X])
	                           +Numerics.square(pointListRef[3*(m-npt)+Y]-pointList[3*j+Y])
	                           +Numerics.square(pointListRef[3*(m-npt)+Z]-pointList[3*j+Z]);
	                }
	                BBt[n/step][m/step] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                      *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=npt;j<npt+nrf;j++) if (j%step!=0) {
	                double distN;
	                if (n<npt) {
	                    distN = Numerics.square(pointList[3*n+X]-pointListRef[3*(j-npt)+X])
	                           +Numerics.square(pointList[3*n+Y]-pointListRef[3*(j-npt)+Y])
	                           +Numerics.square(pointList[3*n+Z]-pointListRef[3*(j-npt)+Z]);
	                } else {
	                    distN = Numerics.square(pointListRef[3*(n-npt)+X]-pointListRef[3*(j-npt)+X])
	                           +Numerics.square(pointListRef[3*(n-npt)+Y]-pointListRef[3*(j-npt)+Y])
	                           +Numerics.square(pointListRef[3*(n-npt)+Z]-pointListRef[3*(j-npt)+Z]);
	                }
	                double distM;
	                if (m<npt) {
	                    distM = Numerics.square(pointList[3*m+X]-pointListRef[3*(j-npt)+X])
	                           +Numerics.square(pointList[3*m+Y]-pointListRef[3*(j-npt)+Y])
	                           +Numerics.square(pointList[3*m+Z]-pointListRef[3*(j-npt)+Z]);
	                } else {   
	                    distM = Numerics.square(pointListRef[3*(m-npt)+X]-pointListRef[3*(j-npt)+X])
	                           +Numerics.square(pointListRef[3*(m-npt)+Y]-pointListRef[3*(j-npt)+Y])
	                           +Numerics.square(pointListRef[3*(m-npt)+Z]-pointListRef[3*(j-npt)+Z]);
	                }
	                BBt[n/step][m/step] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                      *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	        }
	    }
	    	    
        // update banded matrix
        double[][] sqAinvBBt = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            sqAinvBBt[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                sqAinvBBt[n][m] += sqAinv[n][p]*BBt[p][m];
            }
        }
        BBt = null;
        
        double[][] Afull = new double[msize][msize];
        
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            Afull[n][m] = Acore[n][m];
            for (int p=0;p<msize;p++) {
                Afull[n][m] += sqAinvBBt[n][p]*sqAinv[p][m];
            }
        }
        sqAinvBBt = null;
        
        // second eigendecomposition: S = A + A^-1/2 BB^T A^-1/2
        mtx = new Array2DRowRealMatrix(Afull);
        eig = new EigenDecomposition(mtx);
        
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
        embeddingListRef = new float[nrf*ndims];
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npt;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist;
                    if (m<npt) {
                        dist = Numerics.square(pointList[3*n+X]-pointList[3*m+X])
                              +Numerics.square(pointList[3*n+Y]-pointList[3*m+Y])
                              +Numerics.square(pointList[3*n+Z]-pointList[3*m+Z]);
                    } else {
                        dist = Numerics.square(pointList[3*n+X]-pointListRef[3*(m-npt)+X])
                              +Numerics.square(pointList[3*n+Y]-pointListRef[3*(m-npt)+Y])
                              +Numerics.square(pointList[3*n+Z]-pointListRef[3*(m-npt)+Z]);
                    } 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/step][d];
                }
                embeddingList[n+d*npt] = (float)embed;
            }
            for (int n=0;n<nrf;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist;
                    if (m<npt) {
                        dist = Numerics.square(pointListRef[3*n+X]-pointList[3*m+X])
                              +Numerics.square(pointListRef[3*n+Y]-pointList[3*m+Y])
                              +Numerics.square(pointListRef[3*n+Z]-pointList[3*m+Z]);
                    } else {
                        dist = Numerics.square(pointListRef[3*n+X]-pointListRef[3*(m-npt)+X])
                              +Numerics.square(pointListRef[3*n+Y]-pointListRef[3*(m-npt)+Y])
                              +Numerics.square(pointListRef[3*n+Z]-pointListRef[3*(m-npt)+Z]);
                    }
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n+npt]*Vortho[m/step][d];
                }
                embeddingListRef[n+d*nrf] = (float)embed;
            }
        }
        
		return;
	}

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
	    // reset diagonal to zero to have correct distance when closest
	    for (int n=0;n<msize*step;n+=step) {
	        sampledist[n/step][n/step] = 0.0f;
	    }
        float dmax=0.0f, dmean=0.0f;
        for (int n=0;n<msize*step;n+=step) for (int m=0;m<msize*step;m+=step) {
            if (sampledist[n/step][m/step]>dmax) dmax = sampledist[n/step][m/step];
            dmean += sampledist[n/step][m/step];
        }
        dmean /= msize*msize;
        System.out.println("(mean: "+dmean+", max:"+dmax+")");
        
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
                    Azero[n/step][m/step] = 1.0/(1.0+dist/scale);
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
                b0r[n/step] += 1.0/(1.0+dist/scale);
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
                    
                degree[n] += (1.0 + ainvb0r[m/step])*1.0/(1.0+dist/scale);
            }
        }
        System.out.println("build first approximation");
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
            Acore[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {	            
                Acore[n/step][m/step] = -Azero[n/step][m/step]/degree[n];
                Acore[m/step][n/step] = -Azero[m/step][n/step]/degree[m];
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
                BBt[n/step][n/step] += 1.0/(1.0+distN/scale)/degree[j]
                                      *1.0/(1.0+distN/scale)/degree[j];
                
                for (int m=n+step;m<msize*step;m+=step) {
                    double distM = distances[0][j]+sampledist[m/step][closest[0][j]-1];
                
                    for (int d=1;d<depth;d++) if (closest[d][j]>0) {
                        distM = Numerics.min(distM, distances[d][j]+sampledist[m/step][closest[d][j]-1]);
                    }
                    BBt[n/step][m/step] += 1.0/(1.0+distN/scale)/degree[j]
                                          *1.0/(1.0+distM/scale)/degree[j];
                                  
                    BBt[m/step][n/step] += 1.0/(1.0+distN/scale)/degree[j]
                                          *1.0/(1.0+distM/scale)/degree[j];
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
                    
                    embed += 1.0/(1.0+dist/scale)/degree[n]*Vortho[m/step][dim];
                }
                embeddingList[n+dim*npt] = (float)embed;
            }
        }
        
		return;
	}

	public void sparseMeshDistanceEmbedding(float affinity, int depth) {
	    
	    // data size
	    int npt = pointList.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
        
	    // precompute surface-based distances
	    int[][] ngbp =  MeshProcessing.generatePointNeighborTable(npt, triangleList);
	    float avgp = 0.0f;
	    int maxp = 0;
        for (int p=0;p<npt;p++) {
            avgp += ngbp[p].length;
            if (ngbp[p].length>maxp) maxp = ngbp[p].length;
        }
        avgp /= npt;
        System.out.println("Average connectivity: "+avgp+" (max: "+maxp+")");
	    
	    if (depth<avgp) depth = Numerics.ceil(avgp);
	    System.out.println("depth: "+depth);
	    
	    int[] sampleList = new int[npt];
	    for (int n=0;n<msize;n++) {
	        sampleList[n] = n+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, sampleList);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int p=0;p<npt;p++) {
            if (distances[d][p]>maxdist) maxdist = distances[d][p];
        }
        System.out.println("depth: "+depth+", distance propagation max: "+maxdist);
	    
        // build a complete sample distance map? should be doable, roughly O(msize^2)
        float[][] sampledist = new float[msize][msize];
        for (int n=0;n<msize;n++) {
            for (int d=0;d<depth;d++) {
	            if (closest[d][n*step]>0) {
	                sampledist[n][closest[d][n*step]-1] = distances[d][n*step];
	                sampledist[closest[d][n*step]-1][n] = distances[d][n*step];
	            }
	        }
	    }
	    // set to false to skip the propagation
	    boolean missing=true;
	    int nstep=0;
	    int nmiss=-1;
	    int nprev=1;
	    while (missing && (nmiss!=nprev) ) {
            missing=false;
            nprev=nmiss;
            nmiss=0;
            for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
                if (sampledist[n][m]>0) {
                    for (int d=0;d<depth;d++) {
                        if (closest[d][n*step]>0) {
                            if (sampledist[closest[d][n*step]-1][m]==0) sampledist[closest[d][n*step]-1][m] = distances[d][n*step]+sampledist[n][m];
                            else sampledist[closest[d][n*step]-1][m] = Numerics.min(sampledist[closest[d][n*step]-1][m],distances[d][n*step]+sampledist[n][m]);
                            sampledist[m][closest[d][n*step]-1] = sampledist[closest[d][n*step]-1][m];
                        }
                        if (closest[d][m*step]>0) {
                            if (sampledist[closest[d][m*step]-1][n]==0) sampledist[closest[d][m*step]-1][n] = distances[d][m]+sampledist[n][m];
                            else sampledist[closest[d][m*step]-1][n] = Numerics.min(sampledist[closest[d][m*step]-1][n],distances[d][m*step]+sampledist[n][m]);
                            sampledist[n][closest[d][m*step]-1] = sampledist[closest[d][m*step]-1][n];
                        }
                    }
                } else {
                    nmiss++;
                    missing=true;
                }
            }
            nstep++;
        }
        System.out.println("approximate distance propagation: "+nstep+" (missing: "+nmiss+")");
	    // reset diagonal to zero to have correct distance when closest
	    for (int n=0;n<msize;n++) {
	        sampledist[n][n] = 0.0f;
	    }
	    float threshold = scale*(1.0f-affinity)/affinity;
	    System.out.println("distance threshold: "+threshold);
	    int skipped=0;
	    boolean[][] samplemask = new boolean[msize][msize];
	    for (int n=0;n<msize;n++) for (int m=n+1;m<msize;m++) {
	        if (sampledist[n][m]>threshold) {
	            samplemask[n][m]=false;
	            samplemask[m][n]=false;
	            sampledist[n][m]=0.0f;
	            sampledist[m][n]=0.0f;
	            skipped++;
	        }
	    }
        System.out.println("skipped sample connections: "+skipped+" over "+msize*(msize-1)/2);
	    float dmax=0.0f, dmean=0.0f;
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (sampledist[n][m]>dmax) dmax = sampledist[n][m];
            dmean += sampledist[n][m];
        }
        dmean /= msize*msize;
        System.out.println("(mean: "+dmean+", max:"+dmax+")");
        
        if (scale<0) scale = dmean;
        
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[msize][msize];
	    
	    for (int n=0;n<msize;n++) {
	        
	        // self affinitiy should be 1?
            Azero[n][n] = 1.0;
	        
	        for (int m=n+1;m<msize;m++) if (samplemask[n][m]) {	            
                double dist = sampledist[n][m];
                
                if (dist>0) {
                    Azero[n][m] = 1.0/(1.0+dist/scale);
                    Azero[m][n] = Azero[n][m];
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
	    
	    boolean[][] ptmask = new boolean[npt][msize];
	    skipped=0;
	    
	    // first the rows from A, B^T
	    for (int n=0;n<msize;n++) {
	        for (int m=0;m<msize;m++) if (samplemask[n][m]) {	    
	            a0r[n] += Azero[n][m];
	        }
	        for (int p=0;p<npt;p++) if (p%step!=0 || p>=msize*step) {
                double dist = distances[0][p]+sampledist[n][closest[0][p]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][p]>0) {
                    dist = Numerics.min(dist, distances[d][p]+sampledist[n][closest[d][p]-1]);
                }
                if (dist>threshold) {
                    ptmask[p][n] = false;      
                    skipped++;
                } else {
                    b0r[n] += 1.0/(1.0+dist/scale);
                    ptmask[p][n] = true;
                }
            }
	    }
	    System.out.println("skipped other connections: "+skipped+" over "+msize*(npt-msize));
	    
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
	    for (int n=0;n<msize;n++) {
	        degree[n] = a0r[n]+b0r[n];
	    }
        for (int p=0;p<npt;p++) if (p%step!=0 || p>=msize*step) {
            for (int m=0;m<msize;m++) if (ptmask[p][m]) {	
                double dist = distances[0][p]+sampledist[m][closest[0][p]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][p]>0) {
                    dist = Numerics.min(dist, distances[d][p]+sampledist[m][closest[d][p]-1]);
                }
                    
                degree[p] += (1.0 + ainvb0r[m])*1.0/(1.0+dist/scale);
            }
        }
        System.out.println("build first approximation");
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize;n++) {
	        
            Acore[n][n] = 1.0;
	        
	        for (int m=n+1;m<msize;m++) {	            
                Acore[n/step][m] = -Azero[n][m]/degree[n*step];
                Acore[m/step][n] = -Azero[m][n]/degree[m*step];
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
            for (int n=0;n<msize;n++) if (ptmask[j][n]) {
                double distN = distances[0][j]+sampledist[n][closest[0][j]-1];
                
                for (int d=1;d<depth;d++) if (closest[d][j]>0) {
                    distN = Numerics.min(distN, distances[d][j]+sampledist[n][closest[d][j]-1]);
                }   
                BBt[n][n] += 1.0/(1.0+distN/scale)/degree[j]
                            *1.0/(1.0+distN/scale)/degree[j];
                
                for (int m=n+1;m<msize;m++) if (ptmask[j][m]) {
                    double distM = distances[0][j]+sampledist[m][closest[0][j]-1];
                
                    for (int d=1;d<depth;d++) if (closest[d][j]>0) {
                        distM = Numerics.min(distM, distances[d][j]+sampledist[m][closest[d][j]-1]);
                    }
                    BBt[n][m] += 1.0/(1.0+distN/scale)/degree[j]
                                *1.0/(1.0+distM/scale)/degree[j];
                                  
                    BBt[m][n] += 1.0/(1.0+distN/scale)/degree[j]
                                *1.0/(1.0+distM/scale)/degree[j];
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
            for (int p=0;p<npt;p++) {
                double embed=0.0;
                for (int m=0;m<msize;m++) if (ptmask[p][m]) {
                    double dist = distances[0][p]+sampledist[m][closest[0][p]-1];
                
                    for (int d=1;d<depth;d++) if (closest[d][p]>0) {
                        dist = Numerics.min(dist, distances[d][p]+sampledist[m][closest[d][p]-1]);
                    }
                    
                    embed += 1.0/(1.0+dist/scale)/degree[p]*Vortho[m][dim];
                }
                embeddingList[p+dim*npt] = (float)embed;
            }
        }
        
		return;
	}

	public void fastMeshDistanceEmbedding(int depth){
	    
	    // data size
	    int npt = pointList.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
        
	    // precompute surface-based distances
	    int[][] ngbp =  MeshProcessing.generatePointNeighborTable(npt, triangleList);
	    float avgp = 0.0f;
	    int maxp = 0;
        for (int p=0;p<npt;p++) {
            avgp += ngbp[p].length;
            if (ngbp[p].length>maxp) maxp = ngbp[p].length;
        }
        avgp /= npt;
        System.out.println("Average connectivity: "+avgp+" (max: "+maxp+")");
	    
	    //int depth = Numerics.ceil(avgp);
	    if (depth<0) depth = Numerics.ceil(avgp)*Numerics.ceil(avgp);
	    int[] sampleList = new int[npt];
	    for (int n=0;n<msize;n++) {
	        sampleList[n] = n+1;
	    }
	    float[][] distances = new float[depth][npt];
        int[][] closest = new int[depth][npt];
        
        // build the needed distance functions
        MeshProcessing.computeOutsideDistanceFunctions(depth, distances, closest, pointList, triangleList, sampleList);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int p=0;p<npt;p++) {
            if (distances[d][p]>maxdist) maxdist = distances[d][p];
        }
        System.out.println("depth: "+depth+", distance propagation max: "+maxdist);

        // build a complete sample distance map? should be doable, roughly O(msize^2)
        float[][] sampledist = new float[msize][msize];
        int nrel=0;
        for (int n=0;n<msize;n++) {
            for (int d=0;d<depth;d++) {
	            if (closest[d][n*step]>0) {
	                sampledist[n][closest[d][n*step]-1] = distances[d][n*step];
	                sampledist[closest[d][n*step]-1][n] = distances[d][n*step];
	                nrel+=2;
	            }
	        }
	    }
	    System.out.println("initial connections: "+nrel);
	    /*
	    // set to false to skip the propagation
	    boolean missing=true;
	    int nstep=0;
	    int nmiss=-1;
	    int nprev=1;
	    while (missing && (nmiss!=nprev) ) {
            missing=false;
            nprev=nmiss;
            nmiss=0;
            for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
                if (sampledist[n][m]>0) {
                    for (int d=0;d<depth;d++) {
                        if (closest[d][n*step]>0) {
                            if (sampledist[closest[d][n*step]-1][m]==0) sampledist[closest[d][n*step]-1][m] = distances[d][n*step]+sampledist[n][m];
                            else sampledist[closest[d][n*step]-1][m] = Numerics.min(sampledist[closest[d][n*step]-1][m],distances[d][n*step]+sampledist[n][m]);
                            sampledist[m][closest[d][n*step]-1] = sampledist[closest[d][n*step]-1][m];
                        }
                        if (closest[d][m*step]>0) {
                            if (sampledist[closest[d][m*step]-1][n]==0) sampledist[closest[d][m*step]-1][n] = distances[d][m*step]+sampledist[n][m];
                            else sampledist[closest[d][m*step]-1][n] = Numerics.min(sampledist[closest[d][m*step]-1][n],distances[d][m*step]+sampledist[n][m]);
                            sampledist[n][closest[d][m*step]-1] = sampledist[closest[d][m*step]-1][n];
                        }
                    }
                } else {
                    nmiss++;
                    missing=true;
                }
            }
            nstep++;
        }
        System.out.println("approximate distance propagation: "+nstep+" (missing: "+nmiss+")");
        */
	    // reset diagonal to zero to have correct distance when closest
	    for (int n=0;n<msize;n++) {
	        sampledist[n][n] = 0.0f;
	    }
        float dmax=0.0f, dmean=0.0f;
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) if (sampledist[n][m]>0) {
            if (sampledist[n][m]>dmax) dmax = sampledist[n][m];
            dmean += sampledist[n][m];
        }
        dmean /= msize*msize;
        System.out.println("(mean: "+dmean+", max:"+dmax+")");
        
        if (scale<0) scale = dmean;
        
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    
	    // degree is quadratic: replace by approx (needs an extra matrix inversion)
	    double[][] Azero = new double[msize][msize];
	    
	    for (int n=0;n<msize;n++) {
	        
	        // self affinitiy should be 1?
            Azero[n][n] = 1.0;
	        
	        for (int m=n+1;m<msize;m++) {	            
                double dist = sampledist[n][m];
                
                if (dist>0) {
                    Azero[n][m] = 1.0/(1.0+dist/scale);
                    Azero[m][n] = Azero[n][m];
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
	    for (int n=0;n<msize;n++) {
	        for (int m=0;m<msize;m++) {	    
	            a0r[n] += Azero[n][m];
	        }
	    }
        for (int j=0;j<npt;j++) if (j%step!=0 || j>=msize*step) {
            for (int d=0;d<depth;d++) if (closest[d][j]>0) {
                double dist = distances[d][j];
                
                b0r[closest[d][j]-1] += 1.0/(1.0+dist/scale);
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
	    for (int n=0;n<msize;n++) {
	        degree[n*step] = a0r[n]+b0r[n];
	    }
	    for (int j=0;j<npt;j++) if (j%step!=0 || j>=msize*step) {
            for (int d=0;d<depth;d++) if (closest[d][j]>0) {
                double dist = distances[d][j];
                
                degree[j] += (1.0 + ainvb0r[closest[d][j]-1])*1.0/(1.0+dist/scale);
            }
        }
        System.out.println("build first approximation");
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize;n++) {
	        
            Acore[n][n] = 1.0;
	        
	        for (int m=n+1;m<msize;m++) {	            
                Acore[n][m] = -Azero[n][m]/degree[n*step];
                Acore[m][n] = -Azero[m][n]/degree[m*step];
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
            for (int dN=0;dN<depth;dN++) if (closest[dN][j]>0) {
                double distN = distances[dN][j];
                
                BBt[closest[dN][j]-1][closest[dN][j]-1] += 1.0/(1.0+distN/scale)/degree[j]
                                                          *1.0/(1.0+distN/scale)/degree[j];
                                                              
                for (int dM=dN+1;dM<depth;dM++) if (closest[dM][j]>0) {
                    double distM = distances[dM][j];
            
                    BBt[closest[dN][j]-1][closest[dM][j]-1] += 1.0/(1.0+distN/scale)/degree[j]
                                                              *1.0/(1.0+distM/scale)/degree[j];
                              
                    BBt[closest[dM][j]-1][closest[dN][j]-1] += 1.0/(1.0+distN/scale)/degree[j]
                                                              *1.0/(1.0+distM/scale)/degree[j];
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
            for (int n=0;n<msize;n++) {
                double embed=0.0;
                for (int m=0;m<msize;m++) {
                    double dist = sampledist[m][n];
                
                    embed += 1.0/(1.0+dist/scale)/degree[n*step]*Vortho[m][dim];
                }
                embeddingList[n*step+dim*npt] = (float)embed;
            }
            for (int j=0;j<npt;j++) if (j%step!=0 || j>=msize*step) {
                double embed=0.0;
                for (int d=0;d<depth;d++) if (closest[d][j]>0) {
                    double dist = distances[d][j];
                
                    embed += 1.0/(1.0+dist/scale)/degree[j]*Vortho[closest[d][j]-1][dim];
                }
                embeddingList[j+dim*npt] = (float)embed;
            }
            /*
            for (int n=0;n<npt;n++) {
                double embed=0.0;
                for (int d=0;d<depth;d++) if (closest[d][n]>0) {
                    double dist = distances[d][n];
                
                    embed += 1.0/(1.0+dist/scale)/degree[n]*Vortho[closest[d][n]-1][dim];
                }
                embeddingList[n+dim*npt] = (float)embed;
            }
            */
        }
        
		return;
	}


}