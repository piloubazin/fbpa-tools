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
public class SpectralDenseMatrixEmbedding {

    
	// jist containers
    private float[] matrixA;
    private float[] embeddingA;
    private float[] pointsA;
    
    private float[] matrixB;
    private float[] embeddingB;
    private float[] pointsB;
    
	private int ndims = 10;
	private int msize = 800;
	private float scale = 1.0f;
	private float space = 1.0f;
	private float link = 0.1f;
	
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
	public final void setSubjectMatrix(float[] val) { matrixB = val; }
	public final void setSubjectPoints(float[] val) { pointsB = val; }
	
	public final void setReferenceMatrix(float[] val) { matrixA = val; }
	public final void setReferencePoints(float[] val) { pointsA = val; }
	

	public final void setDimensions(int val) { ndims = val; }
	public final void setMatrixSize(int val) { msize = val; }
	public final void setDistanceScale(float val) { scale = val; }
	public final void setSpatialScale(float val) { space = val; }
	public final void setLinkingFactor(float val) { link = val; }
					
	// create outputs
	public final float[] 	getSubjectEmbeddings() { return embeddingB; }
	public final float[] 	getReferenceEmbeddings() { return embeddingA; }

	public void matrixRotatedSpatialEmbedding() {

	    // data size
	    int npa = Numerics.round(FastMath.sqrt(matrixA.length));
	    System.out.println("reference matrix size: "+npa);
	    
        int npb = Numerics.round(FastMath.sqrt(matrixB.length));
	    System.out.println("subject matrix size: "+npb);
	    
	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    //matrixSimpleReferenceEmbedding();
	    matrixReferenceJointEmbedding();
	    float[] refEmbedding = new float[npa*ndims];
	    for (int n=0;n<npa*ndims;n++) {
	        refEmbedding[n] = embeddingA[n];
	    }
	    
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    matrixSpatialJointEmbedding();
	    	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    double[][] rot = new double[ndims][ndims];
	    for (int m=0;m<ndims;m++) for (int n=0;n<ndims;n++) {
	        rot[m][n] = 0.0;
	        for (int i=0;i<npa;i++) {
	            rot[m][n] += embeddingA[i+m*npa]*refEmbedding[i+n*npa];
	        }
	    }
	    float[] rotated = new float[npb*ndims];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int j=0;j<npb;j++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += embeddingB[j+m*npb]*rot[m][n];
	            }
	            rotated[j+n*npb] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int j=0;j<npb;j++) {
	            rotated[j+n*npb] /= norm;
	        }
	    }
	    for (int n=0;n<npb*ndims;n++) {
            embeddingB[n] = rotated[n];
        }
        // for checking: not really needed if all goes well
        System.out.println("-- rotating reference embedding --");
	    rotated = new float[npa*ndims];
	    for (int n=0;n<ndims;n++) {
	        double norm=0.0;
            for (int i=0;i<npa;i++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += embeddingA[i+m*npa]*rot[m][n];
	            }
	            rotated[i+n*npa] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int i=0;i<npa;i++) {
	            rotated[i+n*npa] /= norm;
	        }
	    }
	    for (int n=0;n<npb*ndims;n++) {
            embeddingA[n] = rotated[n];
        }
	}

	public void matrixSimpleJointEmbedding() {
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npa = Numerics.round(FastMath.sqrt(matrixA.length));
	    System.out.println("reference matrix size: "+npa);
	    
        int npb = Numerics.round(FastMath.sqrt(matrixB.length));
	    System.out.println("subject matrix size: "+npb);
        
	    // 1. build the partial representation
	    int stepa = Numerics.floor(npa/msize);
	    int stepb = Numerics.floor(npb/msize);
	    System.out.println("step size: "+stepa+", "+stepb);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[2*msize][2*msize];
	    double[] degree = new double[2*msize]; 
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        
	        // self affinitiy should be 1?
            Azero[n/stepa][n/stepa] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixA[n+m*npa];
	                         
	            Azero[n/stepa][m/stepa] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[m/stepa][n/stepa] = Azero[n/stepa][m/stepa];
                degree[n/stepa] += Azero[n/stepa][m/stepa];
                degree[m/stepa] += Azero[m/stepa][n/stepa];
            }
        }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        
	        // self affinitiy should be 1?
            Azero[msize+n/stepb][msize+n/stepb] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepb;m<msize*stepb;m+=stepb) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixB[n+m*npb];
	                         
	            Azero[msize+n/stepb][msize+m/stepb] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[msize+m/stepb][msize+n/stepb] = Azero[msize+n/stepb][msize+m/stepb];
                degree[msize+n/stepb] += Azero[msize+n/stepb][msize+m/stepb];
                degree[msize+m/stepb] += Azero[msize+m/stepb][msize+n/stepb];
            }
        }
        // off diagonal links: just use indices (may be better to bring the geometry into it)
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        for (int m=0;m<msize*stepb;m+=stepb) {
	            if (n==m) {
                    Azero[n/stepa][msize+m/stepb] = 1.0;
                    Azero[msize+m/stepb][n/stepa] = 1.0;
                
                    degree[n/stepa] += Azero[n/stepa][msize+m/stepb];
                    degree[msize+m/stepb] += Azero[msize+m/stepb][n/stepa];
                }
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
	            double dist = matrixA[n+m*npa];
	            
                Acore[n/stepa][m/stepa] = -Azero[n/stepa][m/stepa]/FastMath.sqrt(degree[n/stepa]*degree[m/stepa]);
                Acore[m/stepa][n/stepa] = -Azero[m/stepa][n/stepa]/FastMath.sqrt(degree[m/stepa]*degree[n/stepa]);
            }
        }  
	    for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepb;m+=stepb) {
                if (n==m) {
                    Acore[n/stepa][msize+m/stepb] = -Azero[n/stepa][msize+m/stepb]/FastMath.sqrt(degree[n/stepa]*degree[msize+m/stepb]);
                    Acore[msize+m/stepb][n/stepa] = -Azero[msize+m/stepb][n/stepa]/FastMath.sqrt(degree[msize+m/stepb]*degree[n/stepa]);
                }
            }
        }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        Acore[msize+n/stepb][msize+n/stepb] = 1.0;
	    }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        for (int m=n+stepb;m<msize*stepb;m+=stepb) {
	            double dist = matrixB[n+m*npb];
	            
	            Acore[msize+n/stepb][msize+m/stepb] = -Azero[msize+n/stepb][msize+m/stepb]/FastMath.sqrt(degree[msize+n/stepb]*degree[msize+m/stepb]);
                Acore[msize+m/stepb][msize+n/stepb] = -Azero[msize+m/stepb][msize+n/stepb]/FastMath.sqrt(degree[msize+m/stepb]*degree[msize+n/stepb]);
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        double[] eigval = new double[ndims+2];
        for (int s=0;s<ndims+2;s++) {
            eigval[s] = eig.getRealEigenvalues()[2*msize-1-s];
        }
        
        double[][] init = new double[ndims+2][npa];
        for (int dim=0;dim<ndims+2;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npa;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = matrixA[n+m*npa];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/stepa,2*msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    init[dim][n] = (sum/den);
                }
            }
        }
        
        embeddingA = new float[npa*ndims];
        for (int dim=2;dim<ndims+2;dim++) {
            double norm=0.0;
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-2)*npa] = (float)(init[dim][n]);
                norm += embeddingA[n+(dim-2)*npa]*embeddingA[n+(dim-2)*npa];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-2)*npa] /= norm;
            }
        }
        
        init = new double[ndims+2][npb];
        for (int dim=0;dim<ndims+2;dim++) {
            //System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npb;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*stepb;m+=stepb) {
                    double dist = matrixB[n+m*npb];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(msize+m/stepb,2*msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    init[dim][n] = (sum/den);
                }
            }
        }
        
        embeddingB = new float[npb*ndims];
        for (int dim=2;dim<ndims+2;dim++) {
            double norm=0.0;
            for (int n=0;n<npb;n++) {
                embeddingB[n+(dim-2)*npb] = (float)(init[dim][n]);
                norm += embeddingB[n+(dim-2)*npb]*embeddingB[n+(dim-2)*npb];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npb;n++) {
                embeddingB[n+(dim-2)*npb] /= norm;
            }
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<npb;m++) prod += embeddingB[m+v1*npb]*embeddingB[m+v2*npb];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
                
		return;
	}

	public void matrixSpatialJointEmbedding() {
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npa = Numerics.round(FastMath.sqrt(matrixA.length));
	    System.out.println("reference matrix size: "+npa);
	    
        int npb = Numerics.round(FastMath.sqrt(matrixB.length));
	    System.out.println("subject matrix size: "+npb);
        
	    // 1. build the partial representation
	    int stepa = Numerics.floor(npa/msize);
	    int stepb = Numerics.floor(npb/msize);
	    System.out.println("step size: "+stepa+", "+stepb);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[2*msize][2*msize];
	    double[] degree = new double[2*msize]; 
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        
	        // self affinitiy should be 1?
            Azero[n/stepa][n/stepa] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixA[n+m*npa];
	                         
	            Azero[n/stepa][m/stepa] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[m/stepa][n/stepa] = Azero[n/stepa][m/stepa];
                degree[n/stepa] += Azero[n/stepa][m/stepa];
                degree[m/stepa] += Azero[m/stepa][n/stepa];
            }
        }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        
	        // self affinitiy should be 1?
            Azero[msize+n/stepb][msize+n/stepb] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepb;m<msize*stepb;m+=stepb) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixB[n+m*npb];
	                         
	            Azero[msize+n/stepb][msize+m/stepb] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[msize+m/stepb][msize+n/stepb] = Azero[msize+n/stepb][msize+m/stepb];
                degree[msize+n/stepb] += Azero[msize+n/stepb][msize+m/stepb];
                degree[msize+m/stepb] += Azero[msize+m/stepb][msize+n/stepb];
            }
        }
        // off diagonal links: full distance matrix
        for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepb;m+=stepb) {
                double dist = Numerics.square(pointsA[3*n+X]-pointsB[3*m+X])
                             +Numerics.square(pointsA[3*n+Y]-pointsB[3*m+Y])
                             +Numerics.square(pointsA[3*n+Z]-pointsB[3*m+Z]);
            
                Azero[n/stepa][msize+m/stepb] = link/(1.0+FastMath.sqrt(dist)/space);
                //Azero[n/stepa][msize+m/stepb] = link*FastMath.exp(-dist/(space*space));
                Azero[msize+m/stepb][n/stepa] = Azero[n/stepa][msize+m/stepb];
        
                degree[n/stepa] += Azero[n/stepa][msize+m/stepb];
                degree[msize+m/stepb] += Azero[msize+m/stepb][n/stepa];
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
	            double dist = matrixA[n+m*npa];
	            
                Acore[n/stepa][m/stepa] = -Azero[n/stepa][m/stepa]/FastMath.sqrt(degree[n/stepa]*degree[m/stepa]);
                Acore[m/stepa][n/stepa] = -Azero[m/stepa][n/stepa]/FastMath.sqrt(degree[m/stepa]*degree[n/stepa]);
            }
        }  
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        Acore[msize+n/stepb][msize+n/stepb] = 1.0;
	    }
	    for (int n=0;n<msize*stepb;n+=stepb) {
	        for (int m=n+stepb;m<msize*stepb;m+=stepb) {
	            double dist = matrixB[n+m*npb];
	            
	            Acore[msize+n/stepb][msize+m/stepb] = -Azero[msize+n/stepb][msize+m/stepb]/FastMath.sqrt(degree[msize+n/stepb]*degree[msize+m/stepb]);
                Acore[msize+m/stepb][msize+n/stepb] = -Azero[msize+m/stepb][msize+n/stepb]/FastMath.sqrt(degree[msize+m/stepb]*degree[msize+n/stepb]);
            }
        }
	    for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepb;m+=stepb) {
                Acore[n/stepa][msize+m/stepb] = -Azero[n/stepa][msize+m/stepb]/FastMath.sqrt(degree[n/stepa]*degree[msize+m/stepb]);
                Acore[msize+m/stepb][n/stepa] = -Azero[msize+m/stepb][n/stepa]/FastMath.sqrt(degree[msize+m/stepb]*degree[n/stepa]);
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        double[] eigval = new double[ndims+2];
        for (int s=0;s<ndims+2;s++) {
            eigval[s] = eig.getRealEigenvalues()[2*msize-1-s];
        }
        
        double[][] initA = new double[ndims+1][npa];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npa;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = matrixA[n+m*npa];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/stepa,2*msize-1-dim);
                    den += val;
                }
                for (int m=0;m<msize*stepb;m+=stepb) {
                    double dist = Numerics.square(pointsA[3*n+X]-pointsB[3*m+X])
                                 +Numerics.square(pointsA[3*n+Y]-pointsB[3*m+Y])
                                 +Numerics.square(pointsA[3*n+Z]-pointsB[3*m+Z]);
                                 
                    double val = link/(1.0+FastMath.sqrt(dist)/space);
	                
	                sum += val*eig.getV().getEntry(msize+m/stepb,2*msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    initA[dim][n] = (sum/den);
                }
            }
            /*
            for (int n=0;n<msize*stepa;n+=stepa) {
                initA[dim][n] = eig.getV().getEntry(n/stepa,2*msize-1-dim);
            }
            */
        }
        
        
        double[][] initB = new double[ndims+1][npb];
        for (int dim=0;dim<ndims+1;dim++) {
            //System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npb;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*stepb;m+=stepb) {
                    double dist = matrixB[n+m*npb];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(msize+m/stepb,2*msize-1-dim);
                    den += val;
                }
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = Numerics.square(pointsB[3*n+X]-pointsA[3*m+X])
                                 +Numerics.square(pointsB[3*n+Y]-pointsA[3*m+Y])
                                 +Numerics.square(pointsB[3*n+Z]-pointsA[3*m+Z]);
                                 
                    double val = link/(1.0+FastMath.sqrt(dist)/space);
	                
	                sum += val*eig.getV().getEntry(m/stepa,2*msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    initB[dim][n] = (sum/den);
                }
            }
            /*
            for (int n=0;n<msize*stepb;n+=stepb) {
                initB[dim][n] = eig.getV().getEntry(msize+n/stepb,2*msize-1-dim);
            }
            */
        }
        
        embeddingA = new float[npa*ndims];
        embeddingB = new float[npb*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            
            double norm=0.0;
            for (int n=0;n<npa;n++) {
                //embeddingA[n+(dim-1)*npa] = (float)(initA[dim][n]/initA[0][n]);
                embeddingA[n+(dim-1)*npa] = (float)(initA[dim][n]);
                norm += embeddingA[n+(dim-1)*npa]*embeddingA[n+(dim-1)*npa];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-1)*npa] /= norm;
            }
            norm=0.0;
            for (int n=0;n<npb;n++) {
                //embeddingB[n+(dim-1)*npb] = (float)(initB[dim][n]/initB[0][n]);
                embeddingB[n+(dim-1)*npb] = (float)(initB[dim][n]);
                norm += embeddingB[n+(dim-1)*npb]*embeddingB[n+(dim-1)*npb];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npb;n++) {
                embeddingB[n+(dim-1)*npb] /= norm;
            }
            /*
            // raw outputs
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-1)*npa] = (float)(initA[dim-1][n]);
            }
            for (int n=0;n<npb;n++) {
                embeddingB[n+(dim-1)*npb] = (float)(initB[dim-1][n]);
            }*/
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<npb;m++) prod += embeddingB[m+v1*npb]*embeddingB[m+v2*npb];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
                
		return;
	}

	public void matrixReferenceJointEmbedding() {
	    // here we simply stack the vectors and use the same distance (spatial coordinates)
	    // for intra- and inter- mesh correspondences
	    
	    // data size
	    int npa = Numerics.round(FastMath.sqrt(matrixA.length));
	    System.out.println("reference matrix size: "+npa);
	    
        // 1. build the partial representation
	    int stepa = Numerics.floor(npa/msize);
	    System.out.println("step size: "+stepa);
	    
	    //build Laplacian / Laplace-Beltrami operator (just Laplace for now)
	    // degree is quadratic: replace by approx (needs an extra matrix inversion
	    double[][] Azero = new double[2*msize][2*msize];
	    double[] degree = new double[2*msize]; 
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        
	        // self affinitiy should be 1?
            Azero[n/stepa][n/stepa] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixA[n+m*npa];
	                         
	            Azero[n/stepa][m/stepa] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[m/stepa][n/stepa] = Azero[n/stepa][m/stepa];
                degree[n/stepa] += Azero[n/stepa][m/stepa];
                degree[m/stepa] += Azero[m/stepa][n/stepa];
            }
        }
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        
	        // self affinitiy should be 1?
            Azero[msize+n/stepa][msize+n/stepa] = 1.0;
	        //degree[n/step] += Azero[n/step][n/step];
	        
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixA[n+m*npa];
	                         
	            Azero[msize+n/stepa][msize+m/stepa] = 1.0/(1.0+FastMath.sqrt(dist)/scale);
                Azero[msize+m/stepa][msize+n/stepa] = Azero[msize+n/stepa][msize+m/stepa];
                degree[msize+n/stepa] += Azero[msize+n/stepa][msize+m/stepa];
                degree[msize+m/stepa] += Azero[msize+m/stepa][msize+n/stepa];
            }
        }
        // off diagonal links: full distance matrix
        for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepa;m+=stepa) {
                double dist = Numerics.square(pointsA[3*n+X]-pointsA[3*m+X])
                             +Numerics.square(pointsA[3*n+Y]-pointsA[3*m+Y])
                             +Numerics.square(pointsA[3*n+Z]-pointsA[3*m+Z]);
            
                Azero[n/stepa][msize+m/stepa] = link/(1.0+FastMath.sqrt(dist)/space);
                //Azero[n/stepa][msize+m/stepb] = link*FastMath.exp(-dist/(space*space));
                Azero[msize+m/stepa][n/stepa] = Azero[n/stepa][msize+m/stepa];
        
                degree[n/stepa] += Azero[n/stepa][msize+m/stepa];
                degree[msize+m/stepa] += Azero[msize+m/stepa][n/stepa];
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
	            double dist = matrixA[n+m*npa];
	            
                Acore[n/stepa][m/stepa] = -Azero[n/stepa][m/stepa]/FastMath.sqrt(degree[n/stepa]*degree[m/stepa]);
                Acore[m/stepa][n/stepa] = -Azero[m/stepa][n/stepa]/FastMath.sqrt(degree[m/stepa]*degree[n/stepa]);
            }
        }  
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        Acore[msize+n/stepa][msize+n/stepa] = 1.0;
	    }
	    for (int n=0;n<msize*stepa;n+=stepa) {
	        for (int m=n+stepa;m<msize*stepa;m+=stepa) {
	            double dist = matrixA[n+m*npa];
	            
	            Acore[msize+n/stepa][msize+m/stepa] = -Azero[msize+n/stepa][msize+m/stepa]/FastMath.sqrt(degree[msize+n/stepa]*degree[msize+m/stepa]);
                Acore[msize+m/stepa][msize+n/stepa] = -Azero[msize+m/stepa][msize+n/stepa]/FastMath.sqrt(degree[msize+m/stepa]*degree[msize+n/stepa]);
            }
        }
	    for (int n=0;n<msize*stepa;n+=stepa) {
            for (int m=0;m<msize*stepa;m+=stepa) {
                Acore[n/stepa][msize+m/stepa] = -Azero[n/stepa][msize+m/stepa]/FastMath.sqrt(degree[n/stepa]*degree[msize+m/stepa]);
                Acore[msize+m/stepa][n/stepa] = -Azero[msize+m/stepa][n/stepa]/FastMath.sqrt(degree[msize+m/stepa]*degree[n/stepa]);
            }
        }
	    // First decomposition: A
        RealMatrix mtx = new Array2DRowRealMatrix(Acore);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        double[] eigval = new double[ndims+1];
        for (int s=0;s<ndims+1;s++) {
            eigval[s] = eig.getRealEigenvalues()[2*msize-1-s];
        }
        
        double[][] initA = new double[ndims+1][npa];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npa;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = matrixA[n+m*npa];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/stepa,2*msize-1-dim);
                    den += val;
                }
                for (int m=0;m<msize*stepa;m+=stepa) {
                    double dist = Numerics.square(pointsA[3*n+X]-pointsA[3*m+X])
                                 +Numerics.square(pointsA[3*n+Y]-pointsA[3*m+Y])
                                 +Numerics.square(pointsA[3*n+Z]-pointsA[3*m+Z]);
                                 
                    double val = link/(1.0+FastMath.sqrt(dist)/space);
	                
	                sum += val*eig.getV().getEntry(msize+m/stepa,2*msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    initA[dim][n] = (sum/den);
                }
            }
        }
               
        embeddingA = new float[npa*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            
            double norm=0.0;
            for (int n=0;n<npa;n++) {
                //embeddingA[n+(dim-1)*npa] = (float)(initA[dim][n]/initA[0][n]);
                embeddingA[n+(dim-1)*npa] = (float)(initA[dim][n]);
                norm += embeddingA[n+(dim-1)*npa]*embeddingA[n+(dim-1)*npa];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-1)*npa] /= norm;
            }
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<npa;m++) prod += embeddingA[m+v1*npa]*embeddingA[m+v2*npa];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
                
		return;
	}
	
	public void matrixSimpleEmbedding() {
	    
	    // data size
	    int npb = Numerics.round(FastMath.sqrt(matrixB.length));
	    System.out.println("subject matrix size: "+npb);
        
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
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixB[n+m*npb];
	                         
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
                    double dist = matrixB[n+m*npb];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/step,msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    init[dim][n] = (sum/den);
                }
            }
        }
        
        embeddingB = new float[npb*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<npb;n++) {
                //embeddingB[n+(dim-1)*npb] = (float)(init[dim][n]/init[0][n]);
                embeddingB[n+(dim-1)*npb] = (float)(init[dim][n]);
                norm += embeddingB[n+(dim-1)*npb]*embeddingB[n+(dim-1)*npb];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npb;n++) {
                embeddingB[n+(dim-1)*npb] /= norm;
            }
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<npb;m++) prod += embeddingB[m+v1*npb]*embeddingB[m+v2*npb];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
        
		return;
	}

	public void matrixSimpleReferenceEmbedding() {
	    
	    // data size
	    int npa = Numerics.round(FastMath.sqrt(matrixA.length));
	    System.out.println("reference matrix size: "+npa);
        
	    // 1. build the partial representation
	    int step = Numerics.floor(npa/msize);
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
	            // we assume the matrix is a symmetric distance measure
	            double dist = matrixA[n+m*npa];
	                         
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
        
        double[][] init = new double[ndims+1][npa];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+dim+": "+eigval[dim]);
            for (int n=0;n<npa;n++) {
                double sum=0.0;
                double den=0.0;
                for (int m=0;m<msize*step;m+=step) {
                    double dist = matrixA[n+m*npa];
	                         
	                double val = 1.0/(1.0+FastMath.sqrt(dist)/scale);
	                
	                sum += val*eig.getV().getEntry(m/step,msize-1-dim);
                    den += val;
                }
                if (den>0) {
                    init[dim][n] = (sum/den);
                }
            }
        }
        
        embeddingA = new float[npa*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-1)*npa] = (float)(init[dim][n]/init[0][n]);
                norm += embeddingA[n+(dim-1)*npa]*embeddingA[n+(dim-1)*npa];
            }
            norm = FastMath.sqrt(norm);
            for (int n=0;n<npa;n++) {
                embeddingA[n+(dim-1)*npa] /= norm;
            }
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<npa;m++) prod += embeddingA[m+v1*npa]*embeddingA[m+v2*npa];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
        
		return;
	}


}