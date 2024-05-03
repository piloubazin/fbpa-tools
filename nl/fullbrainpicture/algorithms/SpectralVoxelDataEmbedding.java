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
public class SpectralVoxelDataEmbedding {

    
	// jist containers
    private float[] inputImage;
    private float[] refImage;
    private float[] ref2imgMapping;
    
    private float[] imgEmbedding;
    private float[] refEmbedding;

    private int nx, ny, nz, nt, nxyz;
	private float rx, ry, rz;

    private int nxr, nyr, nzr, ntr, nxyzr;
	private float rxr, ryr, rzr;

	private float threshold=0.5f;
	
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
	public	static	final    byte	X = 0;
	public	static	final    byte	Y = 1;
	public	static	final    byte	Z = 2;
	public	static	final    byte	T = 3;
	
	// affinity types	
	public	static	final    byte	CAUCHY = 10;
	public	static	final    byte	GAUSS = 20;
	public	static	final    byte	LINEAR = 30;
	private byte affinity_type = LINEAR;
	
	// distance types
	public static final byte    EUCLIDEAN = 11;
	public static final byte    PRODUCT = 22;
	public static final byte    COSINE = 33;
	private byte distance_type = PRODUCT;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setReferenceImage(float[] val) { refImage = val; }
	public final void setReferenceToImageMapping(float[] val) { ref2imgMapping = val; }
	
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
	public final void setDistanceType(String val) { 
	    if (val.equals("Euclidean")) distance_type = EUCLIDEAN;
        else if (val.equals("cosine")) distance_type = COSINE;
        else distance_type = PRODUCT;
	}
					
	public final void setImageDimensions(int x, int y, int z, int t) { nx=x; ny=y; nz=z; nt=t; nxyz=nx*ny*nz; }
	public final void setImageDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2];nt=dim[3]; nxyz=nx*ny*nz; }
	
	public final void setImageResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setImageResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	public final void setReferenceDimensions(int x, int y, int z, int t) { nxr=x; nyr=y; nzr=z; ntr=t; nxyzr=nxr*nyr*nzr; }
	public final void setReferenceDimensions(int[] dim) { nxr=dim[0]; nyr=dim[1]; nzr=dim[2]; ntr=dim[3]; nxyzr=nxr*nyr*nzr; }
	
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
	
	   
    public void rotatedJointSpatialEmbedding(int depth, double alpha) {

	    // make reference embedding
	    System.out.println("-- building reference embedding --");
	    voxelDataReferenceSparseEmbedding(depth, alpha);
	    float[] initEmbedding = new float[refEmbedding.length];
	    for (int n=0;n<refEmbedding.length;n++) {
	        initEmbedding[n] = refEmbedding[n];
	    }
	    
	    // make joint embedding
	    System.out.println("-- building joint embedding --");
	    voxelDataJointSparseEmbedding(depth, alpha);
	    	    
	    // make rotation back into reference space
	    System.out.println("-- rotating joint embedding --");
	    embeddingReferenceRotation(initEmbedding, refEmbedding, imgEmbedding, nxyzr, nxyz, ndims);
	}
	
	public final void voxelDataSparseEmbedding(int depth, double alpha) {
	    int npt=0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (inputImage[xyz]!=0) {
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
	        if (inputImage[xyz]!=0) {
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
	    
	    // build affinities from data distances
        double[][] matrix = new double[msize][msize];
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (distance_type==EUCLIDEAN) {
                for (int t=0;t<nt;t++) {
                    matrix[n][m] += Numerics.square(inputImage[pts[n]+t*nxyz]-inputImage[pts[m]+t*nxyz]);
                }
                matrix[n][m] = FastMath.sqrt(matrix[n][m]);
            } else if (distance_type==PRODUCT) {
                double prodn = 0.0;
                double prodm = 0.0;
                for (int t=0;t<nt;t++) {
                    matrix[n][m] += inputImage[pts[n]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                    prodn += inputImage[pts[n]+t*nxyz]*inputImage[pts[n]+t*nxyz];
                    prodm += inputImage[pts[m]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                }
                if (prodn*prodm>0) {
                    matrix[n][m] = 1.0-matrix[n][m]/FastMath.sqrt(prodn*prodm);
                } else {
                    matrix[n][m] = -1.0;
                }
            } else if (distance_type==COSINE) {
                double prodn = 0.0;
                double prodm = 0.0;
                for (int t=0;t<nt;t++) {
                    matrix[n][m] += inputImage[pts[n]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                    prodn += inputImage[pts[n]+t*nxyz]*inputImage[pts[n]+t*nxyz];
                    prodm += inputImage[pts[m]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                }
                if (prodn*prodm>0) {
                    matrix[n][m] = FastMath.acos(matrix[n][m]/FastMath.sqrt(prodn*prodm));
                } else {
                    matrix[n][m] = -FastMath.PI;
                }
            }
            if (matrix[n][m]>=0) matrix[n][m] = affinity(matrix[n][m]);
            else matrix[n][m] = 0.0;
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
                for (int m=0;m<msize;m++) {
                    double dist=0.0;
                    if (distance_type==EUCLIDEAN) {
                        for (int t=0;t<ntr;t++) {
                            dist += Numerics.square(inputImage[xyz+t*nxyz]-inputImage[pts[m]+t*nxyz]);
                        }
                        dist = FastMath.sqrt(dist);
                    } else if (distance_type==PRODUCT) {
                        double prodn = 0.0;
                        double prodm = 0.0;
                        for (int t=0;t<ntr;t++) {
                            dist += inputImage[xyz+t*nxyz]*inputImage[pts[m]+t*nxyz];
                            prodn += inputImage[xyz+t*nxyz]*inputImage[xyz+t*nxyz];
                            prodm += inputImage[pts[m]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                        }
                        if (prodn*prodm>0) {
                            dist = 1.0-dist/FastMath.sqrt(prodn*prodm);
                        } else {
                            dist = -1.0;
                        }
                    } else if (distance_type==COSINE) {
                        double prodn = 0.0;
                        double prodm = 0.0;
                        for (int t=0;t<ntr;t++) {
                            dist += inputImage[xyz+t*nxyz]*inputImage[pts[m]+t*nxyz];
                            prodn += inputImage[xyz+t*nxyz]*inputImage[xyz+t*nxyz];
                            prodm += inputImage[pts[m]+t*nxyz]*inputImage[pts[m]+t*nxyz];
                        }
                        if (prodn*prodm>0) {
                            dist = FastMath.acos(dist/FastMath.sqrt(prodn*prodm));
                        } else {
                            dist = -FastMath.PI;
                        }
                    }
                    if (dist>=0) {
                        sum += affinity(dist)*eig.getV().getEntry(m,eignum[dim]);
                        den += affinity(dist);
                    }
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
	
	public final void voxelDataJointSparseEmbedding(int depth, double alpha) {
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
        
        // select subject points aligned with reference
        int[] samples = new int[nxyz];
	    int[] pts = new int[msize];
	    
	    for (int n=0;n<msize;n++) {
	        int xs = Numerics.round(ref2imgMapping[prf[n]+X*nxyzr]);
	        int ys = Numerics.round(ref2imgMapping[prf[n]+Y*nxyzr]);
	        int zs = Numerics.round(ref2imgMapping[prf[n]+Z*nxyzr]);
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
	
	public final void voxelDataReferenceSparseEmbedding(int depth, double alpha) {
	    int nrf=0;
	    for (int xyz=0;xyz<nxyzr;xyz++) if (refImage[xyz]!=0) {
	        nrf++;
	    }
        int stpf = Numerics.floor(nrf/msize);
	    System.out.println("step size: "+stpf);
	    
	    int[] samplesRef = new int[nxyzr];
	    int[] prf = new int[msize];
	    int p=0;
	    int s=1;
	    for (int xyz=0;xyz<nxyzr;xyz++) {
	        if (refImage[xyz]!=0) {
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
	    
	    // build affinities from data distances
        double[][] matrixRef = new double[msize][msize];
        for (int n=0;n<msize;n++) for (int m=0;m<msize;m++) {
            if (distance_type==EUCLIDEAN) {
                for (int t=0;t<ntr;t++) {
                    matrixRef[n][m] += Numerics.square(refImage[prf[n]+t*nxyz]-refImage[prf[m]+t*nxyz]);
                }
                matrixRef[n][m] = FastMath.sqrt(matrixRef[n][m]);
            } else if (distance_type==PRODUCT) {
                double prodn = 0.0;
                double prodm = 0.0;
                for (int t=0;t<ntr;t++) {
                    matrixRef[n][m] += refImage[prf[n]+t*nxyz]*refImage[prf[m]+t*nxyz];
                    prodn += refImage[prf[n]+t*nxyz]*refImage[prf[n]+t*nxyz];
                    prodm += refImage[prf[m]+t*nxyz]*refImage[prf[m]+t*nxyz];
                }
                if (prodn*prodm>0) {
                    matrixRef[n][m] = 1.0-matrixRef[n][m]/FastMath.sqrt(prodn*prodm);
                } else {
                    matrixRef[n][m] = -1.0;
                }
            } else if (distance_type==COSINE) {
                double prodn = 0.0;
                double prodm = 0.0;
                for (int t=0;t<ntr;t++) {
                    matrixRef[n][m] += refImage[prf[n]+t*nxyz]*refImage[prf[m]+t*nxyz];
                    prodn += refImage[prf[n]+t*nxyz]*refImage[prf[n]+t*nxyz];
                    prodm += refImage[prf[m]+t*nxyz]*refImage[prf[m]+t*nxyz];
                }
                if (prodn*prodm>0) {
                    matrixRef[n][m] = FastMath.acos(matrixRef[n][m]/FastMath.sqrt(prodn*prodm));
                } else {
                    matrixRef[n][m] = -FastMath.PI;
                }
            }
            if (matrixRef[n][m]>=0) matrixRef[n][m] = affinity(matrixRef[n][m]);
            else matrixRef[n][m] = 0.0;
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
                for (int m=0;m<msize;m++) {
                    double dist=0.0;
                    if (distance_type==EUCLIDEAN) {
                        for (int t=0;t<ntr;t++) {
                            dist += Numerics.square(refImage[xyz+t*nxyzr]-refImage[prf[m]+t*nxyzr]);
                        }
                        dist = FastMath.sqrt(dist);
                    } else if (distance_type==PRODUCT) {
                        double prodn = 0.0;
                        double prodm = 0.0;
                        for (int t=0;t<ntr;t++) {
                            dist += refImage[xyz+t*nxyzr]*refImage[prf[m]+t*nxyzr];
                            prodn += refImage[xyz+t*nxyzr]*refImage[xyz+t*nxyzr];
                            prodm += refImage[prf[m]+t*nxyzr]*refImage[prf[m]+t*nxyzr];
                        }
                        if (prodn*prodm>0) {
                            dist = 1.0-dist/FastMath.sqrt(prodn*prodm);
                        } else {
                            dist = -1.0;
                        }
                    } else if (distance_type==COSINE) {
                        double prodn = 0.0;
                        double prodm = 0.0;
                        for (int t=0;t<ntr;t++) {
                            dist += refImage[xyz+t*nxyz]*refImage[prf[m]+t*nxyz];
                            prodn += refImage[xyz+t*nxyz]*refImage[xyz+t*nxyz];
                            prodm += refImage[prf[m]+t*nxyz]*refImage[prf[m]+t*nxyz];
                        }
                        if (prodn*prodm>0) {
                            dist = FastMath.acos(dist/FastMath.sqrt(prodn*prodm));
                        } else {
                            dist = -FastMath.PI;
                        }
                    }
                    if (dist>=0) {
                        sum += affinity(dist)*eig.getV().getEntry(m,eignum[dim]);
                        den += affinity(dist);
                    }
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

}