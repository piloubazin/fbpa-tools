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
public class SpectralVoxelThicknessEmbedding {

    
	// jist containers
    private float[] inputImage;
    private float[] refImage;
    private float[] refMapping;
    
    private float[] imgEmbedding;
    private float[] refEmbedding;

    private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

    private int nxr, nyr, nzr, nxyzr;
	private float rxr, ryr, rzr;

	private float threshold=0.5f;
	
	private int ndims = 10;
	private int msize = 800;
	private float scale = 1.0f;
	private double space = 1.0f;
	private float link = 1.0f;
	private boolean normalize=true;
	private float ratio=1.0f;
	
	// numerical quantities
	private static final	double	INVSQRT2 = 1.0/FastMath.sqrt(2.0);
	private static final	double	INVSQRT3 = 1.0/FastMath.sqrt(3.0);
	private static final	double	SQRT2 = FastMath.sqrt(2.0);
	private static final	double	SQRT3 = FastMath.sqrt(3.0);

	// direction labeling		
	public	static	final    byte	X = 0;
	public	static	final    byte	Y = 1;
	public	static	final    byte	Z = 2;
	public	static	final    byte	T = 3;
	
	// affinity types	
	public	static	final    byte	CAUCHY = 10;
	public	static	final    byte	GAUSS = 20;
	public	static	final    byte	LINEAR = 30;
	private byte affinity_type = LINEAR;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setReferenceImage(float[] val) { refImage = val; }
	public final void setMapping(float[] val) { refMapping = val; }
	
	public final void setDimensions(int val) { ndims = val; }
	public final void setMatrixSize(int val) { msize = val; }
	public final void setDistanceScale(float val) { scale = val; }
	public final void setSpatialScale(double val) { space = val; }
	public final void setLinkingFactor(float val) { link = val; }
	public final void setAffinityType(String val) { 
	    if (val.equals("Cauchy")) affinity_type = CAUCHY;
        else if (val.equals("Gauss")) affinity_type = GAUSS;
        else affinity_type = LINEAR;
	}
	public final void setThicknessRatio(float val) { ratio = val; }
					
	public final void setImageDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setImageDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setImageResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setImageResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	public final void setReferenceDimensions(int x, int y, int z) { nxr=x; nyr=y; nzr=z; nxyzr=nxr*nyr*nzr; }
	public final void setReferenceDimensions(int[] dim) { nxr=dim[0]; nyr=dim[1]; nzr=dim[2]; nxyzr=nxr*nyr*nzr; }
	
	public final void setReferenceResolutions(float x, float y, float z) { rxr=x; ryr=y; rzr=z; }
	public final void setReferenceResolutions(float[] res) { rxr=res[0]; ryr=res[1]; rzr=res[2]; }
				
	// create outputs
	public final float[] 	getImageEmbedding() { return imgEmbedding; }
	public final float[] 	getReferenceEmbedding() { return refEmbedding; }
	

	private final double affinity(double dist) {
	    //return scale/dist;
	    if (affinity_type==CAUCHY) return 1.0/(1.0+dist*dist/scale/scale);
	    else if (affinity_type==GAUSS) return FastMath.exp(-0.5*dist*dist/scale/scale);
	    else return 1.0/(1.0+dist/scale);
	    //return 1.0/(1.0+dist*dist/(scale*scale));
	}

	private final double linking(double dist) {
	    if (affinity_type==CAUCHY) return link/(1.0+dist*dist/space/space);
	    else if (affinity_type==GAUSS) return link*FastMath.exp(-0.5*dist*dist/space/space);
	    else return link/(1.0+dist/space);
	}
	   
    public void rotatedJointThicknessEmbedding(int depth, double alpha) {

	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    voxelThicknessReferenceSparseEmbedding(depth, alpha);
	    float[] initEmbedding = new float[refEmbedding.length];
	    for (int n=0;n<refEmbedding.length;n++) {
	        initEmbedding[n] = refEmbedding[n];
	    }
	    
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    voxelThicknessJointSparseEmbedding(depth, alpha);
	    	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    embeddingReferenceRotation(initEmbedding, refEmbedding, imgEmbedding, nxyzr, nxyz, ndims);
	}
	
	public final void voxelThicknessSparseEmbedding(int depth, double alpha) {
	    int npt=0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (inputImage[xyz]>threshold) {
	        npt++;
	    }
        System.out.println("region size: "+npt);
        int step = Numerics.floor(npt/msize);
	    System.out.println("step size: "+step);
            
	    int[] samples = new int[nxyz];
	    int[] pts = new int[msize];
	    int p=0;
	    int s=1;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (inputImage[xyz]>threshold) {
                if (p>s*step && s<=msize) {
                    samples[xyz] = s;
                    pts[s-1] = xyz;
                    s++;
                }
                p++;
            } else {
                // mask out regions outside the structure of interest
                samples[xyz] = -1;
            }
	    }
	    
        float[][] distances = new float[depth][nxyz];
        int[][] closest = new int[depth][nxyz];
        
        // compute and add the thickness distance
	    float[] thickness = computeSignedThicknessDistance(inputImage);
	    
	    // build the needed distance functions
        computeOutsideGradientAndDistanceFunctions(depth, distances, closest, samples, thickness, ratio, nx, ny, nz);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int xyz=0;xyz<nxyz;xyz++) {
            if (distances[d][xyz]>maxdist) {
                maxdist = distances[d][xyz];
            }
        }
	    System.out.println("fast marching distances max: "+maxdist);

	    // affinities
        double[][] matrix = distanceMatrixFromVoxelSampling(distances, closest, pts, depth, msize, true);
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
        for (int d=1;d<ndims+1;d++) {
            eigval[d] = 1e16;
            for (int n=0;n<msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[d] && eig.getRealEigenvalues()[n]>eigval[d-1]) {
                    eigval[d] = eig.getRealEigenvalues()[n];
                    eignum[d] = n;
                }
            }
        }
        // tiled results: we should interpolate instead...
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][nxyz];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
            for (int xyz=0;xyz<nxyz;xyz++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closest[d][xyz]>0) {
                    sum += affinity(distances[d][xyz])*eig.getV().getEntry(closest[d][xyz]-1,eignum[dim]);
                    den += affinity(distances[d][xyz]);
                }
                if (den>0) {
                    init[dim][xyz] = (float)(sum/den);
                }
            }
        }

        imgEmbedding = new float[nxyz*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                imgEmbedding[xyz+(dim-1)*nxyz] = (float)(init[dim][xyz]);
                norm += imgEmbedding[xyz+(dim-1)*nxyz]*imgEmbedding[xyz+(dim-1)*nxyz];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int xyz=0;xyz<nxyz;xyz++) {
                imgEmbedding[xyz+(dim-1)*nxyz] /= norm;
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
            for (int xyz=0;xyz<nxyz;xyz++) prod += imgEmbedding[xyz+v1*nxyz]*imgEmbedding[xyz+v2*nxyz];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");
        
        return;
	}
	
	public final void voxelThicknessJointSparseEmbedding(int depth, double alpha) {
	    int nrf=0;
	    for (int xyz=0;xyz<nxyzr;xyz++) if (refImage[xyz]>threshold) {
	        nrf++;
	    }
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step size: "+stpf);
            
	    int[] samplesRef = new int[nxyzr];
	    int[][] prf3 = new int[msize][3];
	    int[] prf = new int[msize];
	    int p=0;
	    int s=1;
	    //for (int xyz=0;xyz<nxyzr;xyz++) {
	    for (int x=0;x<nxr;x++) for (int y=0;y<nyr;y++) for (int z=0;z<nzr;z++) {
	        int xyz = x+nxr*y+nxr*nyr*z;
	        if (refImage[xyz]>threshold) {
                if (p>s*stpf && s<=msize) {
                    samplesRef[xyz] = s;
                    prf3[s-1][X] = x;
                    prf3[s-1][Y] = y;
                    prf3[s-1][Z] = z;
                    prf[s-1] = xyz;
                    s++;
                }
                p++;
            } else {
                // mask out regions outside the structure of interest
                samplesRef[xyz] = -1;
            }
	    }
	    float[][] distancesRef = new float[depth][nxyzr];
        int[][] closestRef = new int[depth][nxyzr];
        
        // build the needed distance functions
        ObjectTransforms.computeOutsideDistanceFunctions(depth, distancesRef, closestRef, samplesRef, nxr, nyr, nzr);
        
        // select subject points aligned with reference
        int[] samples = new int[nxyz];
	    int[] pts = new int[msize];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (inputImage[xyz]<=threshold) {
                // mask out regions outside the structure of interest
                samples[xyz] = -1;
            }
	    }
	    
	    for (int n=0;n<msize;n++) {
	        int xs,ys,zs;
	        if (refMapping==null) {
	            xs = prf3[n][X];
	            ys = prf3[n][Y];
	            zs = prf3[n][Z];
	        } else {
	            int xyzr = prf3[n][X] + nxr*prf3[n][Y] + nxr*nyr*prf3[n][Z];
	            xs = Numerics.round(refMapping[xyzr+X*nxyzr]);
	            ys = Numerics.round(refMapping[xyzr+Y*nxyzr]);
	            zs = Numerics.round(refMapping[xyzr+Z*nxyzr]);
	        }
	        // search for closest neighbor
	        if (inputImage[xs+nx*ys+nx*ny*zs]<=threshold) {
                boolean found = false;
                int delta=1;
                while (!found) {
                    for (int dx=-delta;dx<=delta;dx++) {
                        for (int dy=-delta;dy<=delta;dy++) {
                            for (int dz=-delta;dz<=delta;dz++) {
                                if (dx==-delta || dx==delta || dy==-delta || dy==delta || dz==-delta || dz==delta) {
                                    if (inputImage[xs+dx+nx*(ys+dy)+nx*ny*(zs+dz)]>threshold) {
                                        found = true;
                                        xs = xs+dx;
                                        ys = ys+dy;
                                        zs = zs+dz;
                                        dx=delta+1;
                                        dy=delta+1;
                                        dz=delta+1;
                                    }
                                }
                            }
                        }
                    }
                    delta++;
                    if (delta>=10) {
                        System.out.print("x");
                        found=true;
                    }
                }
            }
            pts[n] = xs+nx*ys+nx*ny*zs;
            samples[xs+nx*ys+nx*ny*zs] = n+1;
        }
        
	    float[][] distances = new float[depth][nxyz];
        int[][] closest = new int[depth][nxyz];
        
        // build the needed distance functions
        ObjectTransforms.computeOutsideDistanceFunctions(depth, distances, closest, samples, nx, ny, nz);
        
        float maxdist = 0.0f;
        for (int d=0;d<depth;d++) for (int xyz=0;xyz<nxyz;xyz++) {
            if (distances[d][xyz]>maxdist) maxdist = distances[d][xyz];
        }
	    float maxdistRef = 0.0f;
        for (int d=0;d<depth;d++) for (int xyz=0;xyz<nxyzr;xyz++) {
            if (distancesRef[d][xyz]>maxdistRef) maxdistRef = distancesRef[d][xyz];
        }
	    System.out.println("fast marching distances max: "+maxdist+", "+maxdistRef);

	    // affinities
        double[][] distmtx = distanceMatrixFromVoxelSampling(distances, closest, pts, depth, msize, true);
        double[][] distmtxRef = distanceMatrixFromVoxelSampling(distancesRef, closestRef, prf, depth, msize, true);
        
        // linking functions
        double[][] linker = new double[msize][msize];
        for (int n=0;n<msize;n++) for (int m=n;m<msize;m++) {
            // linking distance: average of geodesic and point distances?
            // just the geodesics seems most stable.
            // note that it is implied that the sampled points are corresponding across both meshes
            // (which is done via matching above)
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
        for (int d=1;d<ndims+1;d++) {
            eigval[d] = 1e16;
            for (int n=0;n<2*msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[d] && eig.getRealEigenvalues()[n]>eigval[d-1]) {
                    eigval[d] = eig.getRealEigenvalues()[n];
                    eignum[d] = n;
                }
            }
        }
        // from mean coord to neighbors
        double[][] init = new double[ndims+1][nxyz+nxyzr];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
            for (int xyz=0;xyz<nxyz;xyz++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closest[d][xyz]>0) {
                    sum += affinity(distances[d][xyz])*eig.getV().getEntry(closest[d][xyz]-1,eignum[dim]);
                    den += affinity(distances[d][xyz]);
                }
                if (den>0) {
                    init[dim][xyz] = (float)(sum/den);
                }
            }
            for (int xyz=0;xyz<nxyzr;xyz++)  {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closestRef[d][xyz]>0) {
                    sum += affinity(distancesRef[d][xyz])*eig.getV().getEntry(msize+closestRef[d][xyz]-1,eignum[dim]);
                    den += affinity(distancesRef[d][xyz]);
                }
                if (den>0) {
                    init[dim][nxyz+xyz] = (float)(sum/den);
                }
            }
        }

        imgEmbedding = new float[nxyz*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                imgEmbedding[xyz+(dim-1)*nxyz] = (float)(init[dim][xyz]);
                norm += imgEmbedding[xyz+(dim-1)*nxyz]*imgEmbedding[xyz+(dim-1)*nxyz];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int xyz=0;xyz<nxyz;xyz++) {
                imgEmbedding[xyz+(dim-1)*nxyz] /= norm;
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
            for (int xyz=0;xyz<nxyz;xyz++) prod += imgEmbedding[xyz+v1*nxyz]*imgEmbedding[xyz+v2*nxyz];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");

        refEmbedding = new float[nxyzr*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
           for (int xyz=0;xyz<nxyzr;xyz++) {
                refEmbedding[xyz+(dim-1)*nxyzr] = (float)(init[dim][nxyz+xyz]);
                norm += refEmbedding[xyz+(dim-1)*nxyzr]*refEmbedding[xyz+(dim-1)*nxyzr];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int xyz=0;xyz<nxyzr;xyz++) {
                refEmbedding[xyz+(dim-1)*nxyzr] /= norm;
            }
        }
        
		return;
	}
	
	public final void voxelThicknessReferenceSparseEmbedding(int depth, double alpha) {
	    int nrf=0;
	    for (int xyz=0;xyz<nxyzr;xyz++) if (refImage[xyz]>threshold) {
	        nrf++;
	    }
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step size: "+stpf);
	    
	    int[] samplesRef = new int[nxyzr];
	    int[] prf = new int[msize];
	    int p=0;
	    int s=1;
	    for (int xyz=0;xyz<nxyzr;xyz++) {
	        if (refImage[xyz]>threshold) {
                if (p>s*stpf && s<=msize) {
                    samplesRef[xyz] = s;
                    prf[s-1] = xyz;
                    s++;
                }
                p++;
            } else {
                // mask out regions outside the structure of interest
                samplesRef[xyz] = -1;
            }
	    }
	    float[][] distancesRef = new float[depth][nxyzr];
        int[][] closestRef = new int[depth][nxyzr];
        
        // build the needed distance functions
        ObjectTransforms.computeOutsideDistanceFunctions(depth, distancesRef, closestRef, samplesRef, nxr, nyr, nzr);
        
	    float maxdistRef = 0.0f;
        for (int d=0;d<depth;d++) for (int xyz=0;xyz<nxyzr;xyz++) {
            if (distancesRef[d][xyz]>maxdistRef) maxdistRef = distancesRef[d][xyz];
        }
	    System.out.println("fast marching distances max: "+maxdistRef);

	    // affinities
        double[][] matrixRef = distanceMatrixFromVoxelSampling(distancesRef, closestRef, prf, depth, msize, true);
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
        for (int d=1;d<ndims+1;d++) {
            eigval[d] = 1e16;
            for (int n=0;n<msize;n++) {
                if (eig.getRealEigenvalues()[n]<eigval[d] && eig.getRealEigenvalues()[n]>eigval[d-1]) {
                    eigval[d] = eig.getRealEigenvalues()[n];
                    eignum[d] = n;
                }
            }
        }
        // from mean coord to neighbors
        double[][] initRef = new double[ndims+1][nxyzr];
        for (int dim=0;dim<ndims+1;dim++) {
            System.out.println("eigenvalue "+eignum[dim]+": "+eigval[dim]);
            for (int xyz=0;xyz<nxyzr;xyz++) {
                double sum=0.0;
                double den=0.0;
                for (int d=0;d<depth;d++) if (closestRef[d][xyz]>0) {
                    sum += affinity(distancesRef[d][xyz])*eig.getV().getEntry(closestRef[d][xyz]-1,eignum[dim]);
                    den += affinity(distancesRef[d][xyz]);
                }
                if (den>0) {
                    initRef[dim][xyz] = (float)(sum/den);
                }
            }
        }
        
        refEmbedding = new float[nxyzr*ndims];
        for (int dim=1;dim<ndims+1;dim++) {
            double norm=0.0;
            for (int xyz=0;xyz<nxyzr;xyz++) {
                refEmbedding[xyz+(dim-1)*nxyzr] = (float)(initRef[dim][xyz]);
                norm += refEmbedding[xyz+(dim-1)*nxyzr]*refEmbedding[xyz+(dim-1)*nxyzr];
            }
            norm = FastMath.sqrt(norm);
            if (normalize) for (int xyz=0;xyz<nxyzr;xyz++) {
                refEmbedding[xyz+(dim-1)*nxyz] /= norm;
            }
        }
        // copy to the other value for ouput
        imgEmbedding = new float[nxyzr*ndims];
        for (int n=0;n<nxyzr*ndims;n++) {
            imgEmbedding[n] = refEmbedding[n];
        }
        
        // check the result
        System.out.println("orthogonality");
        double mean = 0.0;
        double min = 1e9;
        double max = -1e9;
        double num = 0.0;
        for (int v1=0;v1<ndims-1;v1++) for (int v2=v1+1;v2<ndims;v2++) {
            double prod=0.0;
            for (int xyz=0;xyz<nxyzr;xyz++) prod += imgEmbedding[xyz+v1*nxyzr]*imgEmbedding[xyz+v2*nxyzr];
            mean += prod;
            num++;
            if (prod>max) max = prod;
            if (prod<min) min = prod;
        }
        System.out.println("["+min+" | "+mean/num+" | "+max+"]");

		return;
	}
	
	private final double[][] distanceMatrixFromVoxelSampling(float[][] distances, int[][] closest, int[] pts, int depth, int msize, boolean fullDistance) {
        // precompute surface-based distances
	    double[][] matrix = new double[msize][msize];
	    
	    if (fullDistance) {
            // build a complete sample distance map? should be doable, roughly O(msize^2)
            // very slow for large meshes, though
            float[][] sampledist = new float[msize][msize];
            for (int n=0;n<msize;n++) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][pts[n]]-1);
                    if (m>=0) {
                        sampledist[n][m] = distances[d][pts[n]];
                        sampledist[m][n] = distances[d][pts[n]];
                    }
                }
            }
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
                if (sampledist[n][m]>dmax) dmax = sampledist[n][m];
                if (sampledist[n][m]>0) {
                    dmean += sampledist[n][m];
                    nmean++;
                }
            }
            if (nmean>0) dmean /= nmean;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
    
            // set to false to skip the propagation
            int missing=1;
            int prev = -1;
            int nmiss=0;
            while (missing>0 && missing!=prev) {
                prev = missing;
                missing=0;
                for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
                    if (sampledist[n][m]>0) {
                        for (int d=0;d<depth;d++) {
                            int jm = (closest[d][pts[n]]-1);
                            if (jm>=0) {
                                if (sampledist[jm][m]==0) sampledist[jm][m] = distances[d][pts[n]]+sampledist[n][m];
                                else sampledist[jm][m] = Numerics.min(sampledist[jm][m],distances[d][pts[n]]+sampledist[n][m]);
                                sampledist[m][jm] = sampledist[jm][m];
                            }
                            int jn = (closest[d][pts[m]]-1);
                            if (jn>=0) {
                                if (sampledist[jn][n]==0) sampledist[jn][n] = distances[d][pts[m]]+sampledist[n][m];
                                else sampledist[jn][n] = Numerics.min(sampledist[jn][n],distances[d][pts[m]]+sampledist[n][m]);
                                sampledist[n][jn] = sampledist[jn][n];
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
            for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
                if (sampledist[n][m]>dmax) dmax = sampledist[n][m];
                dmean += sampledist[n][m];
            }
            dmean /= msize*msize;
            System.out.println("(mean: "+dmean+", max:"+dmax+")");
            
            // reset diagonal to zero to have correct distance when closest
            for (int n=0;n<msize;n++) {
                sampledist[n][n] = 0.0f;
            }
            
            if (scale<0) scale = dmean;
            
            for (int n=0;n<msize;n++) {
                for (int m=n+1;m<msize;m++) {
                    double dist = sampledist[n][m];
                    
                    if (dist>0) {
                        matrix[n][m] = dist;
                        matrix[m][n] = dist;
                    }
                }
            }
        } else {
            float dmax=0.0f, dmean=0.0f;
            int nmean=0;
            for (int n=0;n<msize;n++) {
                for (int d=0;d<depth;d++) {
                    int m = (closest[d][pts[n]]-1);
                    if (m>=0) {
                        matrix[n][m] = distances[d][pts[n]];
                        matrix[m][n] = matrix[n][m];
                        
                        if (distances[d][pts[n]]>dmax) dmax = distances[d][pts[n]];
                        if (distances[d][pts[n]]>0) {
                            dmean += distances[d][pts[n]];
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

    private final void embeddingReferenceRotation(float[] ref0, float[] ref1, float[] sub1, int nxyzr, int nxyz, int ndims) {
        
        // build a rotation matrix for reference 1 to 0
        double[] norm0 = new double[ndims];
	    double[] norm1 = new double[ndims];
	    for (int n=0;n<ndims;n++) {
	        norm0[n] = 0.0;
	        norm1[n] = 0.0;
	        for (int i=0;i<nxyzr;i++) {
	            norm0[n] += ref0[i+n*nxyzr]*ref0[i+n*nxyzr];
	            norm1[n] += ref1[i+n*nxyzr]*ref1[i+n*nxyzr];
	        }
	        norm0[n] = FastMath.sqrt(norm0[n]);
	        norm1[n] = FastMath.sqrt(norm1[n]);
	    }
	    double[][] rot = new double[ndims][ndims];
	    for (int m=0;m<ndims;m++) for (int n=0;n<ndims;n++) {
	        rot[m][n] = 0.0;
	        for (int i=0;i<nxyzr;i++) {
	            rot[m][n] += ref1[i+m*nxyzr]/norm1[m]*ref0[i+n*nxyzr]/norm0[n];
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
	    float[] rotated = new float[ndims*nxyz];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int j=0;j<nxyz;j++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += sub1[j+m*nxyz]*rot[m][n];
	            }
	            rotated[j+n*nxyz] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int j=0;j<nxyz;j++) {
	            rotated[j+n*nxyz] /= (float)norm;
	        }
	    }
	    for (int n=0;n<nxyz*ndims;n++) {
	        sub1[n] = rotated[n];
	    }
	    rotated = new float[ndims*nxyzr];
        for (int n=0;n<ndims;n++) {
            double norm=0.0;
            for (int i=0;i<nxyzr;i++) {
	            double val = 0.0;
	            for (int m=0;m<ndims;m++) {
	                val += ref1[i+m*nxyzr]*rot[m][n];
	            }
	            rotated[i+n*nxyzr] = (float)val;
	            norm += val*val;
	        }
	        norm = FastMath.sqrt(norm);
            for (int i=0;i<nxyzr;i++) {
	            rotated[i+n*nxyzr] /= (float)norm;
	        }
	    }
	    for (int n=0;n<nxyzr*ndims;n++) {
	        ref1[n] = rotated[n];
	    }
	    return;
    }

    private float[] computeSignedThicknessDistance(float[] proba) {
		
	    // if needed, convert to levelset
        boolean[] mask = new boolean[nxyz];
        for (int n=0;n<nxyz;n++) mask[n] = true;

		float rmax = Numerics.max(rx,ry,rz);
	    
		float size = 0.0f;
		
		float[] levelset = new float[nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                if (proba[xyz]>=threshold) levelset[xyz] = -1.0f;
                else levelset[xyz] = +1.0f;
                if (proba[xyz]>=0.5f && (proba[xyz+1]<0.5f || proba[xyz-1]<0.5f
                                       || proba[xyz+nx]<0.5f || proba[xyz-nx]<0.5f
                                       || proba[xyz+nx*ny]<0.5f || proba[xyz-nx*ny]<0.5f)) 
                    levelset[xyz] = 0.0f;
            } else levelset[xyz] = +1.0f;
		}
        // second pass to bring the outside at the correct distance
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (proba[xyz]<0.5f && (proba[xyz+1]>=0.5f || proba[xyz-1]>=0.5f
                                   || proba[xyz+nx]>=0.5f || proba[xyz-nx]>=0.5f
                                   || proba[xyz+nx*ny]>=0.5f || proba[xyz-nx*ny]>=0.5f)) 
            
                levelset[xyz] = ObjectTransforms.fastMarchingOutsideNeighborDistance(levelset, xyz, nx,ny,nz, rx,ry,rz);
        }       
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (proba[xyz]>=threshold) size++;
        }
        size = (float)FastMath.cbrt(3.0*size/(4.0*FastMath.PI));
        levelset = ObjectTransforms.fastMarchingDistanceFunction(levelset, size+5.0f, nx, ny, nz, rx, ry, rz);
		
		// compute the gradient norm
		float[] medial = new float[nxyz];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (levelset[xyz]<=0) {
                double grad = 0.0;
                grad += 0.25*Numerics.square( (levelset[xyz+1]-levelset[xyz-1])/(rx/rmax) );
                grad += 0.25*Numerics.square( (levelset[xyz+nx]-levelset[xyz-nx])/(ry/rmax) );
                grad += 0.25*Numerics.square( (levelset[xyz+nx*ny]-levelset[xyz-nx*ny])/(rz/rmax) );
                  
                medial[xyz] = (float)Numerics.max(0.0, 1.0-FastMath.sqrt(grad));
            } else {
                medial[xyz] = 0.0f;
            }
        }
         
		// use to build a distance function
        float[] dist = new float[nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                    if (medial[xyz]>=0.5f) dist[xyz] = -1.0f;
                    else dist[xyz] = +1.0f;
                    if (medial[xyz]>=0.5f && (medial[xyz+1]<0.5f || medial[xyz-1]<0.5f
                                                || medial[xyz+nx]<0.5f || medial[xyz-nx]<0.5f
                                                || medial[xyz+nx*ny]<0.5f || medial[xyz-nx*ny]<0.5f)) 
                        dist[xyz] = 0.0f;
                    
            } else {
                dist[xyz] = +1.0f;
            }
        }
        // second pass to bring the outside at the correct distance
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (medial[xyz]<0.5f && (medial[xyz+1]>=0.5f || medial[xyz-1]>=0.5f
                                        || medial[xyz+nx]>=0.5f || medial[xyz-nx]>=0.5f
                                        || medial[xyz+nx*ny]>=0.5f || medial[xyz-nx*ny]>=0.5f)) 
            
                dist[xyz] = ObjectTransforms.fastMarchingOutsideNeighborDistance(dist, xyz, nx,ny,nz, rx,ry,rz);
        }
        /*
        // flip one side?
        int xyz0 = -1;
        float maxmed = 0.0f;
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (medial[xyz]>=0.5f && (medial[xyz+1]<0.5f || medial[xyz-1]<0.5f
                                                || medial[xyz+nx]<0.5f || medial[xyz-nx]<0.5f
                                                || medial[xyz+nx*ny]<0.5f || medial[xyz-nx*ny]<0.5f))
                if (medial[xyz]>maxmed) {
                    maxmed = medial[xyz];
                    xyz0 = xyz;
                }
        }
        // 
        
        dist = ObjectTransforms.fastMarchingDistanceFunction(dist, size+5.0f, nx, ny, nz, rx, ry, rz);
        
        /* not needed: only use local gradients
        // use max medial gradient to define orientation? need to propagate too?
        double gradxM = 0.0;
        double gradyM = 0.0;
        double gradzM = 0.0;
        float medialmax = 0.5f;
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (medial[xyz]>=medialmax) {
                medialmax = medial[xyz];
                gradxM = Numerics.maxmag(dist[xyz+1]-dist[xyz], dist[xyz]-dist[xyz-1])/(rx/rmax);
                gradyM = Numerics.maxmag(dist[xyz+nx]-dist[xyz], dist[xyz]-dist[xyz-nx])/(ry/rmax);
                gradzM = Numerics.maxmag(dist[xyz+nx*ny]-dist[xyz], dist[xyz]-dist[xyz-nx*ny])/(rz/rmax);
            }
        }
        double norm = FastMath.sqrt(gradxM*gradxM+gradyM*gradyM+gradzM*gradzM);
        gradxM /= norm;
        gradyM /= norm;
        gradzM /= norm;
        
        boolean[] flip = new boolean[nxyz];
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (proba[xyz]>threshold) {
                double prod = gradxM*0.5*( (dist[xyz+1]-dist[xyz-1])/(rx/rmax) )
                             +gradyM*0.5*( (dist[xyz+nx]-dist[xyz-nx])/(ry/rmax) )
                             +gradzM*0.5*( (dist[xyz+nx*ny]-dist[xyz-nx*ny])/(rz/rmax) );
                             
                if (prod<0) flip[xyz] = true;             
                             
            }
        }
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (flip[xyz]) {
                dist[xyz] = -dist[xyz];
            }
        }*/
        return levelset;
	}
	
	private static final void computeOutsideGradientAndDistanceFunctions(int nb, float[][] distances, int[][] closest, int[] labels, float[] field, float scaling, int nx, int ny, int nz)  {
        BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeap2D.MINTREE);
				        		
		// compute the neighboring labels and corresponding distance functions (! not the MGDM functions !)
        //if (debug) System.out.print("fast marching\n");		
        heap.reset();
		// initialize the heap from boundaries
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
        	if (labels[xyz]>0) {
                // search for boundaries (exclude interface with negative regions, only expand labels above zero)
                for (byte k = 0; k<6; k++) {
                    int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
                    if (labels[xyzn]!=labels[xyz] && labels[xyzn]>-1) {
                        // add to the heap
                        heap.addValue((1.0f-scaling)*0.5f+scaling*Numerics.abs(field[xyz]-field[xyzn]),xyzn,labels[xyz]);
                    }
                }
            }
        }
		//if (debug) System.out.print("init\n");		

        // grow the labels and functions
        int[] processed = new int[nx*ny*nz]; // note: using a byte instead of boolean for the second pass
		float[] nbdist = new float[6];
		boolean[] nbflag = new boolean[6];
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int xyz = heap.getFirstId1();
        	int lb = heap.getFirstId2();
			heap.removeFirst();

			// if more than nmgdm labels have been found already, this is done
			if (processed[xyz]>=nb)  continue;
			
			// check if the current label is already accounted for
			boolean found=false;
			for (int n=0;n<processed[xyz];n++) if (closest[n][xyz]==lb) found=true;
			if (found) continue;
			
			// update the distance functions at the current level
			distances[processed[xyz]][xyz] = dist;
			closest[processed[xyz]][xyz] = lb;
			processed[xyz]++; // update the current level
 			
			// find new neighbors
			for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				
				if (labels[xyzn]!=lb && labels[xyzn]>-1) {
                    found=false;
                    if (processed[xyzn]>=nb) { // no point in adding neighbors that are already set, faster
                        found=true;
                    } else {
                        for (int n=0;n<processed[xyzn];n++) if (closest[n][xyzn]==lb) found=true;
                    }
                    if (!found) {// must be in outside the object or its processed neighborhood
                        // compute new distance based on processed neighbors for the same object
                        for (byte l=0; l<6; l++) {
                            nbdist[l] = -1.0f;
                            nbflag[l] = false;
                            int xyznb = ObjectTransforms.fastMarchingNeighborIndex(l, xyzn, nx, ny, nz);
                            // note that there is at most one value used here
                            for (int n=0;n<processed[xyznb];n++) if (closest[n][xyznb]==lb) {
                                nbdist[l] = (1.0f-scaling)*distances[n][xyznb]+scaling*Numerics.abs(field[xyzn]-field[xyznb]);
                                nbflag[l] = true;
                                n = processed[xyznb];
                            }			
                        }
                        float newdist = ObjectTransforms.minimumMarchingDistance(nbdist, nbflag);
                        
                        // add to the heap
                        heap.addValue(newdist,xyzn,lb);
                    }
				}
			}			
		}
       return;
     }


}