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


}