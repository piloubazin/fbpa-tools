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
public class SpectralMeshDenseJointEmbedding {

    
	// jist containers
    private float[] pointListA;
    private int[] triangleListA;
    private float[] embeddingListA;
    
    private float[] pointListB;
    private int[] triangleListB;
    private float[] embeddingListB;

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
	public final void setSurfacePoints(float[] val) { pointListB = val; }
	public final void setSurfaceTriangles(int[] val) { triangleListB = val; }

	public final void setReferencePoints(float[] val) { pointListA = val; }
	public final void setReferenceTriangles(int[] val) { triangleListA = val; }


	public final void setDimensions(int val) { ndims = val; }
	public final void setMatrixSize(int val) { msize = val; }
	public final void setDistanceScale(float val) { scale = val; }
					
	// create outputs
	public final float[] 	getEmbeddingValues() { return embeddingListB; }
	public final float[] 	getReferenceEmbeddingValues() { return embeddingListA; }
	
	public void pointDistanceEmbedding(){
	    
	    // data size
	    int npa = pointListA.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npa/msize);
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
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	                         
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
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	                         
	            a0r[n/step] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	            */
	            a0r[n/step] += Azero[n/step][m/step];
	        }
	        for (int m=0;m<npa;m++) if (m%step!=0 || m>=msize*step) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	            
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
	    double[] degree = new double[npa];
	    /*
	    for (int n=0;n<npa;n++) {
	        degree[n] = 0.0;
	    }*/
	    for (int n=0;n<msize*step;n+=step) {
	        degree[n] = a0r[n/step]+b0r[n/step];
	    }
	    for (int n=0;n<npa;n++) if (n%step!=0 || n>=msize*step) {
	        for (int m=0;m<msize*step;m+=step) {	
                double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
                             +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
                             +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
            
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
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	                         
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
        embeddingList = new float[pointListA.length/3*ndims];
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+eig.getRealEigenvalue(d));
            for (int n=0;n<msize*step;n+=step) {
                embeddingList[n+d*pointListA.length/3] = (float)eig.getEigenvector(d).getEntry(n/step);
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
	            for (int j=0;j<npa;j++) if (j%step!=0 || j>=msize*step) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
	                              
	                double distM = Numerics.square(pointListA[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListA[3*j+Z]);
	                              
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
        embeddingListA = new float[npa*ndims];
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npa;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    
                    double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
                                 +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
                                 +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
                                 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/step][d];
                }
                embeddingListA[n+d*npa] = (float)embed;
            }
        }
        
		return;
	}

	public void pointDistanceReferenceEmbedding(){
	    
	    // data size
	    int npb = pointListB.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npb/msize);
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
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	                         
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
	    for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {	    
	            a0r[n/step] += Azero[n/step][m/step];
	        }
	        for (int m=0;m<npb;m++) if (m%step!=0 || m>=msize*step) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	            
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
	    double[] degree = new double[npb];
	    for (int n=0;n<msize*step;n+=step) {
	        degree[n] = a0r[n/step]+b0r[n/step];
	    }
	    for (int n=0;n<npb;n++) if (n%step!=0 || n>=msize*step) {
	        for (int m=0;m<msize*step;m+=step) {	
                double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
                             +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
                             +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
            
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
        
        for (int n=0;n<msize*step;n+=step) {
	        for (int m=0;m<msize*step;m+=step) {
	            BBt[n/step][m/step] = 0.0;
	            for (int j=0;j<npb;j++) if (j%step!=0 || j>=msize*step) {
	                double distN = Numerics.square(pointListB[3*n+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*n+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*n+Z]-pointListB[3*j+Z]);
	                              
	                double distM = Numerics.square(pointListB[3*m+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*m+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*m+Z]-pointListB[3*j+Z]);
	                              
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
        embeddingListB = new float[npb*ndims];
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npb;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    
                    double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
                                 +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
                                 +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
                                 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/step][d];
                }
                embeddingListB[n+d*npb] = (float)embed;
            }
        }
        
		return;
	}
	
	public void pointDistanceJointEmbedding() {
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npa = pointListA.length/3;
	    int npb = pointListB.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npa/msize+npb/msize);
	    System.out.println("step size: "+step);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    double[] degree = new double[npa+npb];
	    for (int n=0;n<npa;n++) {
	        degree[n] = 0.0;
	        for (int m=0;m<npa;m++) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<npb;m++) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListB[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
	    for (int n=0;n<npb;n++) {
	        degree[n+pointListA.length/3] = 0.0;
	        for (int m=0;m<pointListA.length/3;m++) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListA[3*m+Z]);
	            
	            degree[n+npa] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<npb;m++) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	            
	            degree[n+npa] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
        
	    // square core matrix
	    double[][] Acore = new double[msize][msize];
	    
	    for (int n=0;n<msize*step;n+=step) {
	        
            Acore[n/step][n/step] = 1.0;
	        
	        for (int m=n+step;m<msize*step;m+=step) {
	            double dist;
	            if (n<npa && m<npa) {
	                dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                      +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                      +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	            } else if (n>=npa && m<npa) {
	                dist = Numerics.square(pointListB[3*(n-npa)+X]-pointListA[3*m+X])
	                      +Numerics.square(pointListB[3*(n-npa)+Y]-pointListA[3*m+Y])
	                      +Numerics.square(pointListB[3*(n-npa)+Z]-pointListA[3*m+Z]);
	            } else if (n<npa && m>=npa) {
	                dist = Numerics.square(pointListA[3*n+X]-pointListB[3*(m-npa)+X])
	                      +Numerics.square(pointListA[3*n+Y]-pointListB[3*(m-npa)+Y])
	                      +Numerics.square(pointListA[3*n+Z]-pointListB[3*(m-npa)+Z]);
	            } else {
	                dist = Numerics.square(pointListB[3*(n-npa)+X]-pointListB[3*(m-npa)+X])
	                      +Numerics.square(pointListB[3*(n-npa)+Y]-pointListB[3*(m-npa)+Y])
	                      +Numerics.square(pointListB[3*(n-npa)+Z]-pointListB[3*(m-npa)+Z]);
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
        embeddingList = new float[pointListA.length/3*ndims];
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+eig.getRealEigenvalue(d));
            for (int n=0;n<msize*step;n+=step) {
                embeddingList[n+d*pointListA.length/3] = (float)eig.getEigenvector(d).getEntry(n/step);
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
	            for (int j=0;j<npa;j++) if (j%step!=0) {
	                double distN;
	                if (n<npa) {
	                    distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
	                           +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
	                           +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
	                } else {
	                    distN = Numerics.square(pointListB[3*(n-npa)+X]-pointListA[3*j+X])
	                           +Numerics.square(pointListB[3*(n-npa)+Y]-pointListA[3*j+Y])
	                           +Numerics.square(pointListB[3*(n-npa)+Z]-pointListA[3*j+Z]);
	                }
	                double distM;
	                if (m<npa) {
	                    distM = Numerics.square(pointListA[3*m+X]-pointListA[3*j+X])
	                           +Numerics.square(pointListA[3*m+Y]-pointListA[3*j+Y])
	                           +Numerics.square(pointListA[3*m+Z]-pointListA[3*j+Z]);
	                } else {   
	                    distM = Numerics.square(pointListB[3*(m-npa)+X]-pointListA[3*j+X])
	                           +Numerics.square(pointListB[3*(m-npa)+Y]-pointListA[3*j+Y])
	                           +Numerics.square(pointListB[3*(m-npa)+Z]-pointListA[3*j+Z]);
	                }
	                BBt[n/step][m/step] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                      *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=npa;j<npa+npb;j++) if (j%step!=0) {
	                double distN;
	                if (n<npa) {
	                    distN = Numerics.square(pointListA[3*n+X]-pointListB[3*(j-npa)+X])
	                           +Numerics.square(pointListA[3*n+Y]-pointListB[3*(j-npa)+Y])
	                           +Numerics.square(pointListA[3*n+Z]-pointListB[3*(j-npa)+Z]);
	                } else {
	                    distN = Numerics.square(pointListB[3*(n-npa)+X]-pointListB[3*(j-npa)+X])
	                           +Numerics.square(pointListB[3*(n-npa)+Y]-pointListB[3*(j-npa)+Y])
	                           +Numerics.square(pointListB[3*(n-npa)+Z]-pointListB[3*(j-npa)+Z]);
	                }
	                double distM;
	                if (m<npa) {
	                    distM = Numerics.square(pointListA[3*m+X]-pointListB[3*(j-npa)+X])
	                           +Numerics.square(pointListA[3*m+Y]-pointListB[3*(j-npa)+Y])
	                           +Numerics.square(pointListA[3*m+Z]-pointListB[3*(j-npa)+Z]);
	                } else {   
	                    distM = Numerics.square(pointListB[3*(m-npa)+X]-pointListB[3*(j-npa)+X])
	                           +Numerics.square(pointListB[3*(m-npa)+Y]-pointListB[3*(j-npa)+Y])
	                           +Numerics.square(pointListB[3*(m-npa)+Z]-pointListB[3*(j-npa)+Z]);
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
        embeddingListA = new float[npa*ndims];
        embeddingListB = new float[npb*ndims];
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npa;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist;
                    if (m<npa) {
                        dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
                    } else {
                        dist = Numerics.square(pointListA[3*n+X]-pointListB[3*(m-npa)+X])
                              +Numerics.square(pointListA[3*n+Y]-pointListB[3*(m-npa)+Y])
                              +Numerics.square(pointListA[3*n+Z]-pointListB[3*(m-npa)+Z]);
                    } 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/step][d];
                }
                embeddingListA[n+d*npa] = (float)embed;
            }
            for (int n=0;n<npb;n++) {
                double embed=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist;
                    if (m<npa) {
                        dist = Numerics.square(pointListB[3*n+X]-pointListA[3*m+X])
                              +Numerics.square(pointListB[3*n+Y]-pointListA[3*m+Y])
                              +Numerics.square(pointListB[3*n+Z]-pointListA[3*m+Z]);
                    } else {
                        dist = Numerics.square(pointListB[3*n+X]-pointListB[3*(m-npa)+X])
                              +Numerics.square(pointListB[3*n+Y]-pointListB[3*(m-npa)+Y])
                              +Numerics.square(pointListB[3*n+Z]-pointListB[3*(m-npa)+Z]);
                    }
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n+npa]*Vortho[m/step][d];
                }
                embeddingListB[n+d*npb] = (float)embed;
            }
        }
        
		return;
	}

	public void pointDistanceJointBalancedEmbedding() {
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npb = pointListB.length/3;
	    int npa = pointListA.length/3;
	    
	    // 1. build the partial representation
	    int stepa = Numerics.floor(npa/msize);
	    int stepb = Numerics.floor(npb/msize);
	    System.out.println("step size: "+stepa+", "+stepb);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    double[] degree = new double[npa+npb];
	    for (int n=0;n<npa;n++) {
	        degree[n] = 0.0;
	        for (int m=0;m<npa;m++) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<npb;m++) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListB[3*m+Z]);
	            
	            degree[n] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
	    for (int n=0;n<npb;n++) {
	        degree[n+npa] = 0.0;
	        for (int m=0;m<npa;m++) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListA[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListA[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListA[3*m+Z]);
	            
	            degree[n+npa] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	        for (int m=0;m<npb;m++) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	            
	            degree[n+npa] += 1.0/(1.0+FastMath.sqrt(dist)/scale);
	        }
	    }
        
	    // square core matrix
	    double[][] Acore = new double[2*msize][2*msize];
	    
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        Acore[n/stepa][n/stepa] = 1.0;
	    }
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
                             +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
                             +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
	            
                Acore[n/stepa][m/stepa] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n];
                Acore[m/stepa][n/stepa] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[m];
            }
        }  
	    for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepb;m+=stepb) {
	            double dist = Numerics.square(pointListA[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListA[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListA[3*n+Z]-pointListB[3*m+Z]);
	            
	            Acore[n/stepa][msize+m/stepb] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n];
                Acore[msize+m/stepb][n/stepa] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[m+npa];
            }
        }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        Acore[msize+n/stepb][msize+n/stepb] = 1.0;
	    }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        for (int m=n+stepb;m<msize*stepb;m+=stepb) {
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	            
	            Acore[msize+n/stepb][msize+m/stepb] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n+npa];
                Acore[msize+m/stepb][msize+n/stepb] = -1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[m+npa];
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        // sort eigenvalues: not needed, already done :)

        // build S = A + A^-1/2 BB^T A^-1/2
        double[][] sqAinv = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            sqAinv[n][m] = 0.0;
            for (int p=0;p<2*msize;p++) {
                sqAinv[n][m] += eig.getV().getEntry(n,p)
                                *1.0/FastMath.sqrt(eig.getRealEigenvalue(p))
                                *eig.getV().getEntry(m,p);
            }
        }
        mtx = null;
        eig = null;
        
        // banded matrix is too big, we only compute BB^T
        double[][] BBt = new double[2*msize][2*msize];
        
        for (int n=0;n<msize*stepa;n+=stepa) {
	        for (int m=0;m<msize*stepa;m+=stepa) {
	            BBt[n/stepa][m/stepa] = 0.0;
	            for (int j=0;j<npa;j++) if (j%stepa!=0) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
	                
	                double distM = Numerics.square(pointListA[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListA[3*j+Z]);
	                
	                BBt[n/stepa][m/stepa] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                        *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=0;j<npb;j++) if (j%stepb!=0) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListB[3*j+Z]);
	                
	                double distM = Numerics.square(pointListA[3*m+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListB[3*j+Z]);
	                
	                BBt[n/stepa][m/stepa] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j+npa]
	                                      *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j+npa];
	            }
	        }
	    }
        for (int n=0;n<msize*stepb;n+=stepb) {
	        for (int m=0;m<msize*stepa;m+=stepa) {
	            BBt[msize+n/stepb][m/stepa] = 0.0;
	            for (int j=0;j<npa;j++) if (j%stepa!=0) {
	                double distN = Numerics.square(pointListB[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListB[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListB[3*n+Z]-pointListA[3*j+Z]);
	                
	                double distM = Numerics.square(pointListA[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListA[3*j+Z]);
	                
	                BBt[msize+n/stepb][m/stepa] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                              *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=0;j<npb;j++) if (j%stepb!=0) {
	                double distN = Numerics.square(pointListB[3*n+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*n+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*n+Z]-pointListB[3*j+Z]);
	                
	                double distM = Numerics.square(pointListA[3*m+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListB[3*j+Z]);
	                
	                BBt[msize+n/stepb][m/stepa] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j+npa]
	                                              *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j+npa];
	            }
	        }
	    }
        for (int n=0;n<msize*stepa;n+=stepa) {
	        for (int m=0;m<msize*stepb;m+=stepb) {
	            BBt[n/stepa][msize+m/stepb] = 0.0;
	            for (int j=0;j<npa;j++) if (j%stepa!=0) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
	                
	                double distM = Numerics.square(pointListB[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListB[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListB[3*m+Z]-pointListA[3*j+Z]);
	                
	                BBt[n/stepa][msize+m/stepb] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                              *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=0;j<npb;j++) if (j%stepb!=0) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListB[3*j+Z]);
	                
	                double distM = Numerics.square(pointListB[3*m+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*m+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*m+Z]-pointListB[3*j+Z]);
	                
	                BBt[n/stepa][msize+m/stepb] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j+npa]
	                                              *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j+npa];
	            }
	        }
	    }
        for (int n=0;n<msize*stepb;n+=stepb) {
	        for (int m=0;m<msize*stepb;m+=stepb) {
	            BBt[msize+n/stepb][msize+m/stepb] = 0.0;
	            for (int j=0;j<npa;j++) if (j%stepa!=0) {
	                double distN = Numerics.square(pointListB[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListB[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListB[3*n+Z]-pointListA[3*j+Z]);
	                
	                double distM = Numerics.square(pointListB[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListB[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListB[3*m+Z]-pointListA[3*j+Z]);
	                
	                BBt[msize+n/stepb][msize+m/stepb] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j]
	                                                    *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j];
	            }
	            for (int j=0;j<npb;j++) if (j%stepb!=0) {
	                double distN = Numerics.square(pointListB[3*n+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*n+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*n+Z]-pointListB[3*j+Z]);
	                
	                double distM = Numerics.square(pointListB[3*m+X]-pointListB[3*j+X])
	                              +Numerics.square(pointListB[3*m+Y]-pointListB[3*j+Y])
	                              +Numerics.square(pointListB[3*m+Z]-pointListB[3*j+Z]);
	                
	                BBt[msize+n/stepb][msize+m/stepb] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/degree[j+npa]
	                                                    *1.0/(1.0+FastMath.sqrt(distM)/scale)/degree[j+npa];
	            }
	        }
	    }
	    	    
        // update banded matrix
        double[][] sqAinvBBt = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            sqAinvBBt[n][m] = 0.0;
            for (int p=0;p<2*msize;p++) {
                sqAinvBBt[n][m] += sqAinv[n][p]*BBt[p][m];
            }
        }
        BBt = null;
        
        double[][] Afull = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            Afull[n][m] = Acore[n][m];
            for (int p=0;p<2*msize;p++) {
                Afull[n][m] += sqAinvBBt[n][p]*sqAinv[p][m];
            }
        }
        sqAinvBBt = null;
        
        // second eigendecomposition: S = A + A^-1/2 BB^T A^-1/2
        mtx = new Array2DRowRealMatrix(Afull);
        eig = new EigenDecomposition(mtx);
        
        // final result A^-1/2 V D^-1/2
        double[][] Vortho = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            Vortho[n][m] = 0.0;
            for (int p=0;p<2*msize;p++) {
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
        embeddingListA = new float[npa*ndims];
        embeddingListB = new float[npb*ndims];
        
        for (int d=0;d<ndims;d++) {
            System.out.println("eigenvalue "+(d+1)+": "+evals[d]);
            for (int n=0;n<npa;n++) {
                double embed=0.0;
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = Numerics.square(pointListA[3*n+X]-pointListA[3*m+X])
                                 +Numerics.square(pointListA[3*n+Y]-pointListA[3*m+Y])
                                 +Numerics.square(pointListA[3*n+Z]-pointListA[3*m+Z]);
                    
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[m/stepa][d];
                }
                for (int m=0;m<msize*stepb;m+=stepb) {
                    double dist = Numerics.square(pointListA[3*n+X]-pointListB[3*m+X])
                                 +Numerics.square(pointListA[3*n+Y]-pointListB[3*m+Y])
                                 +Numerics.square(pointListA[3*n+Z]-pointListB[3*m+Z]);
                    
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n]*Vortho[msize+m/stepb][d];
                }
                embeddingListA[n+d*npa] = (float)embed;
            }
            for (int n=0;n<npb;n++) {
                double embed=0.0;
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = Numerics.square(pointListB[3*n+X]-pointListA[3*m+X])
                                 +Numerics.square(pointListB[3*n+Y]-pointListA[3*m+Y])
                                 +Numerics.square(pointListB[3*n+Z]-pointListA[3*m+Z]);
                    
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n+npa]*Vortho[m/stepa][d];
                }
                for (int m=0;m<msize*stepb;m+=stepb) {
                    double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
                                 +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
                                 +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
                     
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/degree[n+npa]*Vortho[msize+m/stepb][d];
                }
                embeddingListB[n+d*npb] = (float)embed;
            }
        }
        
		return;
	}

}