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
	private int subsample = 0;
	
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

	public void pointDistanceJointRotatedEmbedding() {
	    
	    int npa = pointListA.length/3;
	    int npb = pointListB.length/3;
	    
	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    pointDistanceReferenceEmbedding();
	    float[] refEmbedding = new float[npa*ndims];
	    for (int n=0;n<npa*ndims;n++) {
	        refEmbedding[n] = embeddingListA[n];
	    }
	    // check orthonormal property
	    double zeromin = 1e9;
	    double zeromax = -1e9;
	    double onemin = 1e9;
	    double onemax = -1e9;
	    double[][] prod = new double[ndims][ndims];
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        double norm=0.0;
	        for (int i=0;i<npa;i++) {
	            norm += embeddingListA[i+n*npa]*embeddingListA[i+m*npa];
	        }
	        prod[n][m] = norm;
	        if (n!=m) {
                if (norm<zeromin) zeromin = norm;
                if (norm>zeromax) zeromax = norm;
            } else {
                if (norm<onemin) onemin = norm;
                if (norm>onemax) onemax = norm;
            }
        }
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) if (n!=m) {
	        prod[n][m] /= FastMath.sqrt(prod[n][n])*FastMath.sqrt(prod[m][m]);
	    }
        System.out.println("orthonormality: ["+onemin+", "+onemax+"] / ["+zeromin+", "+zeromax+"]");
	    System.out.println("full matrix:");
        for (int n=0;n<ndims;n++) {
            System.out.print("[");
            for (int m=0;m<ndims;m++) System.out.print(" "+prod[n][m]+" ");
            System.out.println("]");
        }
        embeddingListB = embeddingListA;
        
        // randomized sparse function for better orthogonality?
        
        //runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, npt, ndims, init, nconnect, alpha);
        /*
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    pointDistanceJointBalancedEmbedding();
	    
	    // check orthonormal property
	    zeromin = 1e9;
	    zeromax = -1e9;
	    onemin = 1e9;
	    onemax = -1e9;
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        double norm=0.0;
	        for (int i=0;i<npa;i++) {
	            norm += embeddingListA[i+n*npa]*embeddingListA[i+m*npa];
	        }
	        if (n!=m) {
                if (norm<zeromin) zeromin = norm;
                if (norm>zeromax) zeromax = norm;
            } else {
                if (norm<onemin) onemin = norm;
                if (norm>onemax) onemax = norm;
            }
        }
        System.out.println("orthonormality: ["+onemin+", "+onemax+"] / ["+zeromin+", "+zeromax+"]");
	    zeromin = 1e9;
	    zeromax = -1e9;
	    onemin = 1e9;
	    onemax = -1e9;
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        double norm=0.0;
	        for (int j=0;j<npb;j++) {
	            norm += embeddingListB[j+n*npb]*embeddingListB[j+m*npb];
	        }
	        if (n!=m) {
                if (norm<zeromin) zeromin = norm;
                if (norm>zeromax) zeromax = norm;
            } else {
                if (norm<onemin) onemin = norm;
                if (norm>onemax) onemax = norm;
            }
        }
        System.out.println("orthonormality: ["+onemin+", "+onemax+"] / ["+zeromin+", "+zeromax+"]");
	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    double[][] rot = new double[ndims][ndims];
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        rot[n][m] = 0.0;
	        for (int i=0;i<npa;i++) {
	            rot[n][m] += refEmbedding[i+n*npa]*embeddingListA[i+m*npa];
	        }
	    }
	    float[] rotated = new float[npb*ndims];
	    for (int j=0;j<npb;j++) {
	        for (int n=0;n<ndims;n++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += rot[n][m]*embeddingListB[j+m*npb];
	            }
	            rotated[j+n*npb] = (float)val;
	        }
	    }
	    for (int n=0;n<npb*ndims;n++) {
            embeddingListB[n] = rotated[n];
        }
        // for checking: not really needed if all goes well
        System.out.println("-- rotating reference embedding --");
	    rotated = new float[npa*ndims];
	    for (int i=0;i<npa;i++) {
	        for (int n=0;n<ndims;n++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += rot[n][m]*embeddingListA[i+m*npa];
	            }
	            rotated[i+n*npa] = (float)val;
	        }
	    }
	    for (int n=0;n<npb*ndims;n++) {
            embeddingListA[n] = rotated[n];
        }

        // check orthonormal property
	    zeromin = 1e9;
	    zeromax = -1e9;
	    onemin = 1e9;
	    onemax = -1e9;
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        double norm=0.0;
	        for (int i=0;i<npa;i++) {
	            norm += embeddingListA[i+n*npa]*embeddingListA[i+m*npa];
	        }
	        if (n!=m) {
                if (norm<zeromin) zeromin = norm;
                if (norm>zeromax) zeromax = norm;
            } else {
                if (norm<onemin) onemin = norm;
                if (norm>onemax) onemax = norm;
            }
        }
        System.out.println("orthonormality: ["+onemin+", "+onemax+"] / ["+zeromin+", "+zeromax+"]");
	    zeromin = 1e9;
	    zeromax = -1e9;
	    onemin = 1e9;
	    onemax = -1e9;
	    for (int n=0;n<ndims;n++) for (int m=0;m<ndims;m++) {
	        double norm=0.0;
	        for (int j=0;j<npb;j++) {
	            norm += embeddingListB[j+n*npb]*embeddingListB[j+m*npb];
	        }
	        if (n!=m) {
                if (norm<zeromin) zeromin = norm;
                if (norm>zeromax) zeromax = norm;
            } else {
                if (norm<onemin) onemin = norm;
                if (norm>onemax) onemax = norm;
            }
        }
        System.out.println("orthonormality: ["+onemin+", "+onemax+"] / ["+zeromin+", "+zeromax+"]");
        */
	}
	
	public void pointDistanceReferenceEmbedding() {
	    
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

        System.out.println("estimate degree");

        double[][] Ainv = new double[msize][msize];
	    for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            Ainv[n][m] = 0.0;
            for (int p=0;p<msize;p++) {
                Ainv[n][m] += eig.getEigenvector(p).getEntry(n)
                                *1.0/eig.getRealEigenvalue(p)
                                *eig.getEigenvector(p).getEntry(m);
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
	            Acore[n/step][m/step] = -Azero[n/step][m/step]/FastMath.sqrt(degree[n]*degree[m]);
                Acore[m/step][n/step] = -Azero[m/step][n/step]/FastMath.sqrt(degree[m]*degree[n]);
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
                sqAinv[n][m] += eig.getEigenvector(p).getEntry(n)
                                *1.0/FastMath.sqrt(eig.getRealEigenvalue(p))
                                *eig.getEigenvector(p).getEntry(m);
            }
        }
        mtx = null;
        eig = null;

        System.out.print(".");        

        // banded matrix is too big, we only compute BB^T
        double[][] BBt = new double[msize][msize];
        
        for (int n=0;n<msize*step;n+=step) {
            BBt[n/step][n/step] = 0.0;
            for (int j=0;j<npa;j++) if (j%step!=0 || j>=msize*step) {
                double distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
                              
                BBt[n/step][n/step] += 1.0/(1.0+FastMath.sqrt(distN)/scale)/FastMath.sqrt(degree[n]*degree[j])
                                      *1.0/(1.0+FastMath.sqrt(distN)/scale)/FastMath.sqrt(degree[n]*degree[j]);
            }
	        for (int m=n+1;m<msize*step;m+=step) {
	            BBt[n/step][m/step] = 0.0;
	            for (int j=0;j<npa;j++) if (j%step!=0 || j>=msize*step) {
	                double distN = Numerics.square(pointListA[3*n+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*n+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*n+Z]-pointListA[3*j+Z]);
	                              
	                double distM = Numerics.square(pointListA[3*m+X]-pointListA[3*j+X])
	                              +Numerics.square(pointListA[3*m+Y]-pointListA[3*j+Y])
	                              +Numerics.square(pointListA[3*m+Z]-pointListA[3*j+Z]);
	                
	                double val = 1.0/(1.0+FastMath.sqrt(distN)/scale)/FastMath.sqrt(degree[n]*degree[j])
	                            *1.0/(1.0+FastMath.sqrt(distM)/scale)/FastMath.sqrt(degree[m]*degree[j]);             
	                BBt[n/step][m/step] += val;
	                BBt[m/step][n/step] += val;
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
                                *eig.getEigenvector(p).getEntry(m)
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
                                 
                    embed += 1.0/(1.0+FastMath.sqrt(dist)/scale)/FastMath.sqrt(degree[n]*degree[m])*Vortho[d][m/step];
                }
                embeddingListA[n+d*npa] = (float)embed;
            }
        }
        
		return;
	}

	public void pointDistanceEmbedding(){
	    
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
	    int npa = pointListA.length/3;
	    int npb = pointListB.length/3;
	    
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
        
        System.out.println("build first approximation");
        
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

        System.out.println("build orthogonalization");
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
        
        System.out.print(".");        

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
	    System.out.print("..");        
	    	    
        // update banded matrix
        double[][] sqAinvBBt = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            sqAinvBBt[n][m] = 0.0;
            for (int p=0;p<2*msize;p++) {
                sqAinvBBt[n][m] += sqAinv[n][p]*BBt[p][m];
            }
        }
        BBt = null;
        
	    System.out.print("...");        

        double[][] Afull = new double[2*msize][2*msize];
        
        for (int n=0;n<2*msize;n++) for (int m=0;m<2*msize;m++) {
            Afull[n][m] = Acore[n][m];
            for (int p=0;p<2*msize;p++) {
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

	public void pointDistanceSparseEmbedding() {
	    
	    // data size
	    int npb = pointListB.length/3;
	    
	    // 1. build the partial representation
	    int step = Numerics.floor(npb/msize);
	    System.out.println("step size: "+step);
        
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[msize][msize];
	    double[] degree = new double[msize]; 
	    for (int n=0;n<msize*step;n+=step) {
	        
	        // self affinitiy should be 1?
            Azero[n/step][n/step] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+step;m<msize*step;m+=step) {	            
	            // for now: approximate geodesic distance with Euclidean distance
	            // note that it is not an issue for data-based distance methods
	            double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                         +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                         +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	                         
	            Azero[n/step][m/step] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[m/step][n/step] = Azero[n/step][m/step];
                degree[n/step] += Azero[n/step][m/step];
                degree[m/step] += Azero[m/step][n/step];
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
	            Acore[n/step][m/step] = -Azero[n/step][m/step]/FastMath.sqrt(degree[n/step]*degree[m/step]);
                Acore[m/step][n/step] = -Azero[m/step][n/step]/FastMath.sqrt(degree[m/step]*degree[n/step]);
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);

        double[] eigval = new double[ndims+1];
        for (int s=0;s<ndims+1;s++) {
            eigval[s] = eig.getRealEigenvalues()[msize-1-s];
        }
        
        double[][] init = new double[ndims+1][npb];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npb;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist = Numerics.square(pointListB[3*n+X]-pointListB[3*m+X])
	                             +Numerics.square(pointListB[3*n+Y]-pointListB[3*m+Y])
	                             +Numerics.square(pointListB[3*n+Z]-pointListB[3*m+Z]);
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/step,msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    init[dim][n] = (sum/den);
                }
            }
        }
        /*
        // refine the result with eigenGame
        System.out.println("Eigen game base volume: "+npb);
                
        // build correlation matrix
        int nmtx = npb*ndims;
        
        System.out.println("non-zero components: "+nmtx);
            
        double[] mtxval = new double[nmtx];
        int[] mtxid1 = new int[nmtx];
        int[] mtxid2 = new int[nmtx];
        ArrayList<Integer>[] mtxinv = new ArrayList[npb];
        for (int p=0;p<npb;p++) mtxinv[p] = new ArrayList<Integer>();
        
        for (int n=0;n<nmtx;n++) {
            // draw random samples from the full matrix
            mtxid1[n] = Numerics.min(Numerics.floor(npb*FastMath.random()),npb-1);
            mtxid2[n] = Numerics.min(Numerics.floor(npb*FastMath.random()),npb-1);
            
            double dist = dist = Numerics.square(pointListB[3*mtxid1[n]+X]-pointListB[3*mtxid2[n]+X])
	                            +Numerics.square(pointListB[3*mtxid1[n]+Y]-pointListB[3*mtxid2[n]+Y])
	                            +Numerics.square(pointListB[3*mtxid1[n]+Z]-pointListB[3*mtxid2[n]+Z]);
                    
            mtxval[n] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
            
            mtxinv[mtxid1[n]].add(n);
            mtxinv[mtxid2[n]].add(n);
        }
        int[][] mtinv = new int[npb][];
        for (int p=0;p<npb;p++) {
            mtinv[p] = new int[mtxinv[p].size()];
            for (int n=0;n<mtxinv[p].size();n++) mtinv[p][n] = (int)mtxinv[p].get(n);
        }
        mtxinv = null;
        
        System.out.println("..correlations");
            
        // get initial vector guesses from subsampled data
        double[] norm = new double[ndims];
        for (int dim=0;dim<ndims;dim++) {
            for (int n=0;n<npb;n++) {
                norm[dim] += init[dim][n]*init[dim][n];
            }
        }
        // rescale to ||V||=1
        for (int i=0;i<ndims;i++) {
            norm[i] = FastMath.sqrt(norm[i]);
            for (int vi=0;vi<npb;vi++) {
                init[i][vi] /= norm[i];
            }
        }
        // or rescale to ratio of lowest?
        for (int i=1;i<ndims+1;i++) {
            for (int vi=0;vi<npb;vi++) {
                init[i-1][vi] = init[i][vi]/init[0][vi];
            }
        }
            
        runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtinv, nmtx, npb, ndims, init, 0.0);
        */
        
        embeddingListB = new float[npb*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            for (int n=0;n<npb;n++) {
                embeddingListB[n+(dim-1)*npb] = (float)(init[dim][n]/init[0][n]);
            }
        }
        
		return;
	}

	// use eigengame with random subsampling to get better eigenvectors ?
	private final void runSparseLaplacianEigenGame(double[] mtval, int[] mtid1, int[] mtid2, int[][] mtinv, int nn0, int nm, int nv, double[][] init, double alpha) {
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
                for (int c=0;c<mtinv[n].length;c++) {
                    if (mtid1[mtinv[n][c]]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[n][c]]/deg[n]*vect[vi][mtid2[mtinv[n][c]]];
                        //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid2[mtinv[c][n]-1]];
                    } else if (mtid2[mtinv[n][c]]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[n][c]]/deg[n]*vect[vi][mtid1[mtinv[n][c]]];
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
                    for (int c=0;c<mtinv[n].length;c++) {
                        if (mtid1[mtinv[n][c]]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[n][c]]/deg[n]*vect[vi][mtid2[mtinv[n][c]]];
                            //Mv[vi][n] += mtval[mtinv[c][n]-1]/(deg[n]*mtw[n])*vect[vi][mtid2[mtinv[c][n]-1]];
                        } else if (mtid2[mtinv[n][c]]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[n][c]]/deg[n]*vect[vi][mtid1[mtinv[n][c]]];
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

}