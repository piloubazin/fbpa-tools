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
	private double space = 1.0f;
	private float link = 1.0f;
	private boolean normalize=true;
	
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
	public final void setSpatialScale(double val) { space = val; }
	public final void setLinkingFactor(float val) { link = val; }
					
	// create outputs
	public final float[] 	getEmbeddingValues() { return embeddingList; }
	public final float[] 	getReferenceEmbeddingValues() { return embeddingListRef; }
	

	private final double affinity(double dist) {
	    //return scale/dist;
	    return 1.0/(1.0+dist/scale);
	    //return 1.0/(1.0+dist*dist/(scale*scale));
	}

	private final double linking(double dist) {
	    //return scale/dist;
	    return link/(1.0+dist/space);
	    //return 1.0/(1.0+dist*dist/(scale*scale));
	}
	   
    public void rotatedJointSpatialEmbedding(int depth, double alpha) {

	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    meshDistanceReferenceSparseEmbedding(depth, alpha);
	    float[] refEmbedding = new float[embeddingListRef.length];
	    for (int n=0;n<embeddingListRef.length;n++) {
	        refEmbedding[n] = embeddingListRef[n];
	    }
	    
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    meshDistanceJointSparseEmbedding(depth, alpha);
	    	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    int npt=pointList.length/3;
        int nrf=pointListRef.length/3;
        embeddingReferenceRotation(refEmbedding, embeddingListRef, embeddingList, nrf, npt, ndims);
	}
	
	public final void meshDistanceSparseEmbedding(int depth, double alpha) {
	    int npt=pointList.length/3;
        int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
            
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
        double[][] matrix = distanceMatrixFromMeshSampling(distances, closest, depth, step, msize, true);
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (matrix[n][m]>0) matrix[n][m] = affinity(matrix[n][m]);
        }
        
        // build Laplacian
        buildLaplacian(matrix, msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
	    // add a dimension for lowest eigenvector
        double[] eigval = new double[ndims+1];
        int[] eignum = new int[ndims+1];
        eigval[0] = 1e16;
        for (int n=0;n<msize;n++) {
            if (eig.getRealEigenvalues()[n]<eigval[0]) {
                eigval[0] = eig.getRealEigenvalues()[n];
                eignum[0] = n;
            }
        }
        for (int s=1;s<ndims+1;s++) {
            eigval[s] = 1e16;
            for (int n=0;n<msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[s] && eig.getRealEigenvalues()[n]>eigval[s-1]) {
                    eigval[s] = eig.getRealEigenvalues()[n];
                    eignum[s] = n;
                }
            }
        }
        // tiled results: we should interpolate instead...
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][npt];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
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
        }

        embeddingList = new float[npt*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]);
                norm += embeddingList[n+(dim-1)*npt]*embeddingList[n+(dim-1)*npt];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] /= norm;
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
            for (int m=0;m<npt;m++) prod += embeddingList[m+v1*npt]*embeddingList[m+v2*npt];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");

		return;
	}
	
	public final void meshDistanceJointSparseEmbedding(int depth, double alpha) {
	    int npt=pointList.length/3;
        int step = Numerics.floor(npt/msize);
        
	    int nrf=pointListRef.length/3;
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step sizes: "+step+", "+stpf);
            
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
        double[][] distmtx = distanceMatrixFromMeshSampling(distances, closest, depth, step, msize, true);
        double[][] distmtxRef = distanceMatrixFromMeshSampling(distancesRef, closestRef, depth, stpf, msize, true);
        
        // linking functions
        double[][] linker = new double[msize][msize];
        for (int n=0;n<msize;n++) for (int m=n;m<msize;m++) {
            // linking distance: average of geodesic and point distances?
            // just the geodesics seems most stable.
            //double distN = FastMath.sqrt(Numerics.square(pointList[3*n+X]-pointListRef[3*n+X])
            //                            +Numerics.square(pointList[3*n+Y]-pointListRef[3*n+Y])
            //                            +Numerics.square(pointList[3*n+Z]-pointListRef[3*n+Z]));

            //double distM = FastMath.sqrt(Numerics.square(pointList[3*m+X]-pointListRef[3*m+X])
            //                            +Numerics.square(pointList[3*m+Y]-pointListRef[3*m+Y])
            //                            +Numerics.square(pointList[3*m+Z]-pointListRef[3*m+Z]));

            // note that it is implied that the sampled points are corresponding across both meshes
            // (not always true!)
            linker[n][m] = 0.5*(distmtx[n][m]+distmtxRef[n][m]);
            linker[m][n] = linker[n][m];
        }
        
        double[][] matrix = new double[2*msize][2*msize];
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            matrix[n][m] = affinity(distmtx[n][m]);
            matrix[n][m+msize] = linking(linker[n][m]);
            matrix[m+msize][n] = linking(linker[m][n]);
            matrix[n+msize][m+msize] = affinity(distmtxRef[n][m]);
        }
        
        // build Laplacian
        buildLaplacian(matrix, 2*msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
	    // add a dimension for lowest eigenvector
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
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
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

        embeddingList = new float[npt*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] = (float)(init[dim][n]);
                norm += embeddingList[n+(dim-1)*npt]*embeddingList[n+(dim-1)*npt];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int n=0;n<npt;n++) {
                embeddingList[n+(dim-1)*npt] /= norm;
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
            for (int m=0;m<npt;m++) prod += embeddingList[m+v1*npt]*embeddingList[m+v2*npt];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");

        embeddingListRef = new float[nrf*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<nrf;n++) {
                embeddingListRef[n+(dim-1)*nrf] = (float)(init[dim][npt+n]);
                norm += embeddingListRef[n+(dim-1)*nrf]*embeddingListRef[n+(dim-1)*nrf];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int n=0;n<nrf;n++) {
                embeddingListRef[n+(dim-1)*nrf] /= norm;
            }
        }
        
		return;
	}
	
	public final void meshDistanceReferenceSparseEmbedding(int depth, double alpha) {
	    int nrf=pointListRef.length/3;
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step size: "+stpf);
            
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
        double[][] matrixRef = distanceMatrixFromMeshSampling(distancesRef, closestRef, depth, stpf, msize, true);
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (matrixRef[n][m]>0) matrixRef[n][m] = affinity(matrixRef[n][m]);
        }
        
        // build Laplacian
        buildLaplacian(matrixRef, msize, alpha);
            
        // SVD? no, eigendecomposition (squared matrix)
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrixRef);
        EigenDecomposition eig = new EigenDecomposition(mtx);
            
        // add a dimension for lowest eigenvector
	    double[] eigval = new double[ndims+1];
        int[] eignum = new int[ndims+1];
        eigval[0] = 1e16;
        for (int n=0;n<msize;n++) {
            if (eig.getRealEigenvalues()[n]<eigval[0]) {
                eigval[0] = eig.getRealEigenvalues()[n];
                eignum[0] = n;
            }
        }
        for (int s=1;s<ndims+1;s++) {
            eigval[s] = 1e16;
            for (int n=0;n<msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[s] && eig.getRealEigenvalues()[n]>eigval[s-1]) {
                    eigval[s] = eig.getRealEigenvalues()[n];
                    eignum[s] = n;
                }
            }
        }
        // from mean coord to neighbors
        double[][] initRef = new double[ndims+1][nrf];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
            for (int n=0;n<nrf;n++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closestRef[d][n]>0) {
                    sum += affinity(distancesRef[d][n])*eig.getV().getEntry(closestRef[d][n]-1,eignum[dim]);
                    den += affinity(distancesRef[d][n]);
                }
                if (den>0) {
                    initRef[dim][n] = (float)(sum/den);
                }
            }
        }
        
        embeddingListRef = new float[nrf*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int n=0;n<nrf;n++) {
                embeddingListRef[n+(dim-1)*nrf] = (float)(initRef[dim][n]);
                norm += embeddingListRef[n+(dim-1)*nrf]*embeddingListRef[n+(dim-1)*nrf];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int n=0;n<nrf;n++) {
                embeddingListRef[n+(dim-1)*nrf] /= norm;
            }
        }
        // copy to the other value for ouput
        embeddingList = new float[nrf*ndims];
        for (int n=0;n<nrf*ndims;n++) {
            embeddingList[n] = embeddingListRef[n];
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int m=0;m<nrf;m++) prod += embeddingList[m+v1*nrf]*embeddingList[m+v2*nrf];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");

		return;
	}
	
	private final double[][] distanceMatrixFromMeshSampling(float[][] distances, int[][] closest, int depth, int step, int msize, boolean fullDistance) {
        // precompute surface-based distances
	    double[][] matrix = new double[msize][msize];
	    
	    if (fullDistance) {
            // build a complete sample distance map? should be doable, roughly O(msize^2)
            // very slow for large meshes, though
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
	            rot[m][n] += ref1[i+m*nrf]/norm1[m]*ref0[i+n*nrf]/norm0[n];
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
	    for (int n=0;n<npt*ndims;n++) {
	        sub1[n] = rotated[n];
	    }
	    rotated = new float[ndims*nrf];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int i=0;i<nrf;i++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += ref1[i+m*nrf]*rot[m][n];
	            }
	            rotated[i+n*nrf] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int i=0;i<nrf;i++) {
	            rotated[i+n*nrf] /= (float)norm;
	        }
	    }
	    for (int n=0;n<nrf*ndims;n++) {
	        ref1[n] = rotated[n];
	    }
	    return;
    }

}