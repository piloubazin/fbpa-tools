package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SuperVoxelSegmentation {

	// jist containers
	private float[] inputImage=null;
	private int[] maskImage=null;
	
	private int[] initsegImage=null;
	private float[][] priorImage=null;
	private int nprior=0;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int nsx, nsy, nsz, nsxyz;
	
	private float scaling;
	private float noise;
	private int nngb = 26;
	private int maxiter = 10;
	private float maxdiff = 0.01f;
	private float threshold = 0.5f;

	private float[] parcelImage;
	private int[] segImage;
	private float[] memsImage;

	private float[] boundaries;
	private float[] cnr;
	private float[] sharpness;

	// intermediate results
	private int[] parcel;
	private float[] rescaled;
	private int[] count;
	private boolean[] mask;
	private float[] levelset;
	private int[] neighbor;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);

	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static	final	byte	Z = 2;
	public	static	final	byte	T = 3;

	// computation variables
	private boolean[][][] obj = new boolean[3][3][3];
	private CriticalPointLUT lut;
	private String	lutdir = null;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }

	public final void setPriorSegmentationImage(int[] val) { initsegImage = val; }
	public final void setMaxPriorImage(float[] val) { 
	    nprior=1;
	    priorImage = new float[nprior][];
	    priorImage[0] = val; 
	}
	public final void setPriorSegmentationNumber(int val) { 
	    nprior = val;
	    priorImage = new float[nprior][];
	}
	public final void setPriorImageAt(int id, float[] val) {
	    priorImage[id] = val;
	}
	    
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	public final void setScalingFactor(float val) { scaling = val; }
	public final void setNoiseLevel(float val) { noise = val; }
	public final void setMaxIterations(int val) { maxiter = val; }
	public final void setMaxDifference(float val) { maxdiff = val; }
	public final void setThreshold(float val) { threshold = val; }
	
	// create outputs
	public final int[] getSegmentationImage() { return segImage; }
	public final float[] getMaxPosteriorImage() { return memsImage; }
	
	public final float[] getParcelImage() { return parcelImage; }
	
	public final int[] getScaledDims() { int[] dims = {nsx, nsy, nsz}; return dims; }

	public void execute(){
	    
	    // make mask
	    mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = true;
		    if (inputImage[xyz]==0) mask[xyz] = false;
		    if (maskImage!=null && maskImage[xyz]==0) mask[xyz] = false;
		}
		maskImage = null;
	    
        System.out.println("Build super-voxel grid");
		supervoxelParcellation();

        System.out.println("Grow boundaries");
        growBoundaries();
        
        System.out.println("Compute posterior segmentation");
        //computeParcelSegmentation();
        computeParcelAggregation(threshold);
	}
	
	public void supervoxelParcellation() {
	    // Compute the supervoxel grid
	    System.out.println("original dimensions: ("+nx+", "+ny+", "+nz+")");
	    nsx = Numerics.floor(nx/scaling);
	    nsy = Numerics.floor(ny/scaling);
	    nsz = Numerics.floor(nz/scaling);
	    nsxyz = nsx*nsy*nsz;
	    System.out.println("rescaled dimensions: ("+nsx+", "+nsy+", "+nsz+")");
	    
	    // init downscaled images
	    parcel = new int[nxyz];
	    rescaled = new float[nsxyz];
	    //memsImage = new float[nsxyz];
	    count = new int[nsxyz];
	    
	    // init supervoxel centroids
	    // include all supervoxels with non-zero values inside
		float[][] centroid = new float[3][nsxyz];
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        centroid[X][xyzs] = 0.0f;
	        centroid[Y][xyzs] = 0.0f;
	        centroid[Z][xyzs] = 0.0f;
	        count[xyzs] = 0;
	        for (int dx=0;dx<scaling;dx++) for (int dy=0;dy<scaling;dy++) for (int dz=0;dz<scaling;dz++) {
	            int xyz = Numerics.floor(xs*scaling)+dx+nx*(Numerics.floor(ys*scaling)+dy)+nx*ny*(Numerics.floor(zs*scaling)+dz);
	            if (mask[xyz]) {
                    centroid[X][xyzs] += Numerics.floor(xs*scaling) + dx;
                    centroid[Y][xyzs] += Numerics.floor(ys*scaling) + dy;
                    centroid[Z][xyzs] += Numerics.floor(zs*scaling) + dz;
                    count[xyzs]++;
                }
            }
            if (count[xyzs]>0) {
                centroid[X][xyzs] /= count[xyzs];
                centroid[Y][xyzs] /= count[xyzs];
                centroid[Z][xyzs] /= count[xyzs];
	        }
	    }
	    
	    // init: search for voxel with lowest gradient within the region instead? (TODO)
	    // OR voxel most representative?
	    double[] selection = new double[27];
	    Percentile median = new Percentile();
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        int x0 = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y0 = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z0 = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz0 = x0+nx*y0+nx*ny*z0;
	        
	        int s=0;
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            selection[s] = inputImage[xyz0+dx+nx*dy+nx*ny*dz];
	            s++;
	        }
	        double med = median.evaluate(selection, 50.0);    
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            if (inputImage[xyz0+dx+nx*dy+nx*ny*dz]==med) {
                    centroid[X][xyzs] = x0+dx;
                    centroid[Y][xyzs] = y0+dy;
                    centroid[Z][xyzs] = z0+dz;
                    dx=2;dy=2;dz=2;
	            }
	        }
	    }
	    // Estimate approximate min,max from sampled grid values
	    float Imin = 1e9f;
	    float Imax = -1e9f;
        for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        
		    int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            if (inputImage[xyz]<Imin) Imin = inputImage[xyz];
	            if (inputImage[xyz]>Imax) Imax = inputImage[xyz];
	        }
	    }
	    // normalize the noise parameter by intensity, but not by distance (-> same speed indep of scale)
	    System.out.println("intensity scale: ["+Imin+", "+Imax+"]");
	    if (Imax>Imin) {
	        noise = noise*noise*(Imax-Imin)*(Imax-Imin);
	    }
	    
	    // start a voxel heap at each center
	    BinaryHeap4D heap = new BinaryHeap4D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MINTREE);
		boolean[] processed = new boolean[nx*ny*nz];
		for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        count[xyzs]=0;
	        
	        int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            // set as starting point
	            parcel[xyz] = xyzs+1;
	            rescaled[xyzs] = inputImage[xyz];
	            count[xyzs] = 1;
	            processed[xyz] = true;
	            
	            // add neighbors to the tree
	            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	                if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                        int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);
                         // exclude zero as mask
                        if (mask[xyznb]) {
                        
                            // distance function
                            float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                        +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                        +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                    
                            float contrast = (inputImage[xyznb]-rescaled[xyzs])
                                            *(inputImage[xyznb]-rescaled[xyzs]);
                                        
                            heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                        }
                    }                    
	            }
	        }
	    }
	    // grow to 
        while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float curr = heap.getFirst();
        	int x = heap.getFirstX();
        	int y = heap.getFirstY();
        	int z = heap.getFirstZ();
        	int xyzs = heap.getFirstK()-1;
        	heap.removeFirst();
        	int xyz = x+nx*y+nx*ny*z;
        	
			if (processed[xyz])  continue;
			
        	// update the cluster
			parcel[xyz] = xyzs+1;
            rescaled[xyzs] = count[xyzs]*rescaled[xyzs] + inputImage[xyz];
            
            centroid[X][xyzs] = count[xyzs]*centroid[X][xyzs] + x;
	        centroid[Y][xyzs] = count[xyzs]*centroid[Y][xyzs] + y;
	        centroid[Z][xyzs] = count[xyzs]*centroid[Z][xyzs] + z;
	        
	        count[xyzs] += 1;
	        rescaled[xyzs] /= count[xyzs];
	        centroid[X][xyzs] /= count[xyzs];
	        centroid[Y][xyzs] /= count[xyzs];
	        centroid[Z][xyzs] /= count[xyzs];
	        
	        processed[xyz]=true;
			
            // add neighbors to the tree
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                    int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);

                    // exclude zero as mask
                    if (mask[xyznb] && !processed[xyznb]) {
                    
                        // distance function
                        float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                    +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                    +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                
                        float contrast = (inputImage[xyznb]-rescaled[xyzs])
                                        *(inputImage[xyznb]-rescaled[xyzs]);
                                        
                        heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                    }
                }
            }
		}
		// output the image of parcels
		parcelImage = new float[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        parcelImage[xyz] = rescaled[parcel[xyz]-1];
		}
	}
	
	public void growBoundaries() {
	    // assume the super voxel step has been run

        // computation variables
        levelset = new float[nxyz]; // note: using a byte instead of boolean for the second pass
		neighbor = new int[nxyz]; // note: using a byte instead of boolean for the second pass
		boolean[] processed = new boolean[nxyz]; // note: using a byte instead of boolean for the second pass
		boolean[] cmask = new boolean[nxyz]; // note: using a byte instead of boolean for the second pass
		float[] nbdist = new float[6];
		boolean[] nbflag = new boolean[6];
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MINTREE);
				        		
		// compute the neighboring labels and corresponding distance functions (! not the MGDM functions !)
        //if (debug) System.out.print("fast marching\n");		
        heap.reset();
        // initialize mask and processing domain
		float maxlvl = Numerics.max(nx/2.0f,ny/2.0f,nz/2.0f);
		for (int x=0; x<nx; x++) for (int y=0; y<ny; y++) for (int z = 0; z<nz; z++) {
			int xyz = x+nx*y+nx*ny*z;
        	levelset[xyz] = 0.5f;
        	
			if (!mask[xyz]) cmask[xyz] = false;
			else if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) cmask[xyz] = true;
			else cmask[xyz] = false;
			if (!cmask[xyz]) levelset[xyz] = maxlvl;
		}
		// initialize the heap from boundaries
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
        	if (cmask[xyz]) {
        	    // search for boundaries
                for (byte k = 0; k<6; k++) {
                    int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
                    if (cmask[xyzn] && parcel[xyz]>0 && parcel[xyzn]>0 && parcel[xyzn]!=parcel[xyz]) {
                        // we assume the levelset value is correct at the boundary
					
                        // add to the heap with previous value
                        heap.addValue(Numerics.abs(levelset[xyzn]),xyzn,xyz);
                    }
                }
            }
        }
		//if (debug) System.out.print("init\n");		

        // grow the labels and functions
        float maxdist = 0.0f;
        while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int xyz = heap.getFirstId1();
        	int ngb = heap.getFirstId2();
			heap.removeFirst();

			// if more than nmgdm labels have been found already, this is done
			if (processed[xyz])  continue;
			
			// update the distance functions at the current level
			levelset[xyz] = dist;
			neighbor[xyz] = parcel[ngb];
			processed[xyz]=true; // update the current level
 			
			// find new neighbors
			for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				
				// must be in outside the object or its processed neighborhood
				if (cmask[xyzn] && !processed[xyzn]) if (parcel[xyzn]==parcel[xyz]) {
					// compute new distance based on processed neighbors for the same object
					for (byte l=0; l<6; l++) {
						nbdist[l] = -1.0f;
						nbflag[l] = false;
						int xyznb = ObjectTransforms.fastMarchingNeighborIndex(l, xyzn, nx, ny, nz);
						// note that there is at most one value used here
						if (cmask[xyznb] && processed[xyznb]) if (parcel[xyznb]==parcel[xyz]) {
							nbdist[l] = Numerics.abs(levelset[xyznb]);
							nbflag[l] = true;
						}			
					}
					float newdist = ObjectTransforms.minimumMarchingDistance(nbdist, nbflag);
					
					// add to the heap
					heap.addValue(newdist,xyzn,ngb);
				}
			}			
		}
		return;
	}
	
	public void fitBasicBoundarySigmoid() {
	    // we assume we have the distance to the boundary and the label of the closest supervoxel
	    // now we fit a sigmoid to intensity, distance for each boundary
	    // the sigmoid has three parameters: height (contrast), slope (sharpness) and offset (true boundary)
	    
	    // we assume a probabilistic prior separation into 3 regions, with the boundary being centered
	    // at zero and the slope being two voxels in spread
	    // thus inside is 1-exp(-|lvl|/d0), and boundary is exp(-|lvl|/d0)
	    
	    float[] interior = new float[nsxyz];
	    float[] incount = new float[nsxyz];
	    
	    float delta = 0.001f;
	    float delta0 = 0.1f;
	    float dist0 = 2.0f;
	    //int nngb = 30;
	    
	    // use HashSets to count the number of needed neighbors per location
	    HashSet[] ngbset = new HashSet[nsxyz];
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        ngbset[xyzs] = new HashSet(nngb);
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        ngbset[label-1].add(ngblb);
	    }
	    // count the number of neighbors per region??
	    // must build a full list...
	    int[][] ngblist = new int[nsxyz][];
	    int[][] bdcount = new int[nsxyz][];
	    float[][] slope = new float[nsxyz][];
	    float[][] dist = new float[nsxyz][];
	    float[][] fitness = new float[nsxyz][];

	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	         ngblist[xyzs] = new int[ngbset[xyzs].size()];
	         bdcount[xyzs] = new int[ngbset[xyzs].size()];
	         slope[xyzs] = new float[ngbset[xyzs].size()];
	         dist[xyzs] = new float[ngbset[xyzs].size()];
	         fitness[xyzs] = new float[ngbset[xyzs].size()];
	    }
	    
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        boolean found=false;
	        int last = ngblist[label-1].length;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                found=true;
	                n=ngblist[label-1].length;
	            } else if (ngblist[label-1][n]==0) {
	                ngblist[label-1][n] = ngblb;
	                n=ngblist[label-1].length;
	            }
	        }
	    }
	    
	    // count numbers of voxels in bdcount: exclude in the rest regions with too few voxels
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	    
            // find the neighbor in the list
	        int loc = -1;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                 loc=n;
	                 n=ngblist[label-1].length;
	            }
	        }
	        if (loc>-1) {
	            bdcount[label-1][loc] += 1.0f;
            }                      
            // also in the neighbor's for negative values? yes, so increase the quality of the estimation
            loc = -1;
	        for (int n=0;n<ngblist[ngblb-1].length;n++) {
	             if (label==ngblist[ngblb-1][n]) {
	                 loc=n;
	                 n=ngblist[ngblb-1].length;
	             }
	        }
	        if (loc>-1) {
                bdcount[ngblb-1][loc] += 1.0f;
            }
        }
        float mincount = scaling;
	    
	    // boundaries: use all the interior of a given region
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        interior[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*inputImage[xyz];
	        incount[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)));
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) interior[xyzs] /= incount[xyzs];
	    }
	    
	    	    
	    // offset: assume it's zero

	    // slope
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        	        
            float wgty = Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta)
                        *Numerics.bounded((interior[label-1]-inputImage[xyz])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta);
            float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)), delta, 1.0f-delta);
                
	        // find the neighbor in the list
	        int loc = -1;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                 loc=n;
	                 n=ngblist[label-1].length;
	            }
	        }
	        if (loc>-1 && bdcount[label-1][loc]>mincount) {
	            slope[label-1][loc] += (2.0f*Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta)-1.0f)
                                            *wgtx*wgty;
                    
                dist[label-1][loc] += levelset[xyz]*wgtx*wgty;
            }                      
            // also in the neighbor's for negative values? yes, so increase the quality of the estimation
            // ut you have to flip the relationship so that the average moves away from zero (for the distance average, which would then become unstable)
            loc = -1;
	        for (int n=0;n<ngblist[ngblb-1].length;n++) {
	             if (label==ngblist[ngblb-1][n]) {
	                 loc=n;
	                 n=ngblist[ngblb-1].length;
	             }
	        }
	        if (loc>-1 && bdcount[ngblb-1][loc]>mincount) {
	            slope[ngblb-1][loc] += -(2.0f*Numerics.bounded((inputImage[xyz]-interior[label-1])/(interior[ngblb-1]-interior[label-1]), delta, 1.0f-delta)-1.0f)
                                            *wgtx*wgty;
                                            
                dist[ngblb-1][loc] += levelset[xyz]*wgtx*wgty;
            }
            
        }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<ngblist[xyzs].length;c++) {
	        if (dist[xyzs][c]>0) slope[xyzs][c] = 0.5f*slope[xyzs][c]/dist[xyzs][c];
	        if (slope[xyzs][c]>2.0f) slope[xyzs][c] = 2.0f;
	        if (slope[xyzs][c]<-2.0f) slope[xyzs][c] = -2.0f;
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        // find the neighbor in the list
	        int loc = -1;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                 loc=n;
	                 n=ngblist[label-1].length;
	            }
	        }
	        if (loc>-1 && bdcount[label-1][loc]>mincount) {
	            if (interior[label-1]>interior[ngblb-1]) {
	                slope[label-1][loc] = Numerics.abs(slope[label-1][loc]);
	            } else {
	                slope[label-1][loc] = -Numerics.abs(slope[label-1][loc]);
	            }
	        }
            // also in the neighbor's for negative values? yes, so increase the quality of the estimation
            loc = -1;
	        for (int n=0;n<ngblist[ngblb-1].length;n++) {
	             if (label==ngblist[ngblb-1][n]) {
	                 loc=n;
	                 n=ngblist[ngblb-1].length;
	             }
	        }
	        if (loc>-1 && bdcount[ngblb-1][loc]>mincount) {
	            if (interior[label-1]>interior[ngblb-1]) {
	                slope[ngblb-1][loc] = -Numerics.abs(slope[ngblb-1][loc]);
	            } else {
	                slope[ngblb-1][loc] = Numerics.abs(slope[ngblb-1][loc]);
	            }
	        }
	    }
	    
	    // estimate goodness of fit
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	    
            // find the neighbor in the list
	        int loc = -1;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                 loc=n;
	                 n=ngblist[label-1].length;
	            }
	        }
	        if (loc>-1 && bdcount[label-1][loc]>mincount) {
	            float sigmoid = Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta);
	            float estimate = 1.0f/(1.0f+(float)FastMath.exp(-slope[label-1][loc]*(levelset[xyz]/dist0)));
	            fitness[label-1][loc] += Numerics.square(sigmoid - estimate);
	            //bdcount[label-1][loc] += 1.0f;
            }                      
            // also in the neighbor's for negative values? yes, so increase the quality of the estimation
            loc = -1;
	        for (int n=0;n<ngblist[ngblb-1].length;n++) {
	             if (label==ngblist[ngblb-1][n]) {
	                 loc=n;
	                 n=ngblist[ngblb-1].length;
	             }
	        }
	        if (loc>-1 && bdcount[ngblb-1][loc]>mincount) {
                float sigmoid = Numerics.bounded((interior[label-1]-inputImage[xyz])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta);
	            float estimate = 1.0f/(1.0f+(float)FastMath.exp(-slope[ngblb-1][loc]*(-levelset[xyz]/dist0)));
	            fitness[ngblb-1][loc] += Numerics.square(sigmoid - estimate);
	            //bdcount[ngblb-1][loc] += 1.0f;
            }
        }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<ngblist[xyzs].length;c++) {
	        if (bdcount[xyzs][c]>mincount) fitness[xyzs][c] /= bdcount[xyzs][c];
	    }
	    float Fmean=0.0f;
	    float Fmax=0.0f;
	    int nf=0;
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<ngblist[xyzs].length;c++) {
	        if (bdcount[xyzs][c]>mincount) {
	            Fmean += fitness[xyzs][c];
	            nf++;
	            if (fitness[xyzs][c]>Fmax) Fmax=fitness[xyzs][c];
	        }
	    }
	    System.out.println("Normalized average error: "+Fmean/nf+", maximum error: "+Fmax);
	    
 	    
	    // boundary SNR
	    float[] noise = new float[nsxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        noise[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*Numerics.square(inputImage[xyz]-interior[label-1]);
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) noise[xyzs] = (float)FastMath.sqrt(noise[xyzs]/incount[xyzs]);
	    }
	    
	    cnr = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
            int label = parcel[xyz];
            int ngblb = neighbor[xyz];
                
            if (levelset[xyz]<1.0f) {
                cnr[xyz] = 2.0f*Numerics.abs(interior[label-1]-interior[ngblb-1])/Numerics.max(delta,noise[label-1]+noise[ngblb-1]);
            }
        }
        
        // sharpness given by slope coefficient
        sharpness = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
            int label = parcel[xyz];
            int ngblb = neighbor[xyz];
                
            if (levelset[xyz]<1.0f) {
                // find the neighbor in the list
                int loc = -1;
                for (int n=0;n<ngblist[label-1].length;n++) {
                     if (ngblb==ngblist[label-1][n]) {
                         loc=n;
                         n=ngblist[label-1].length;
                     }
                }
                if (loc>-1 && bdcount[label-1][loc]>mincount)
                    sharpness[xyz] = Numerics.abs(slope[label-1][loc]);
            }
        }
        
        // boundary probability based on CNR, number of samples, offset value, slope value
        boundaries = new float[nxyz];   
        
        // boundary probability based on Jensen-Shannon divergence? better measure
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
            int label = parcel[xyz];
            int ngblb = neighbor[xyz];
                
            if (levelset[xyz]<1.0f) {
                // find the neighbor in the list
                int loc = -1;
                for (int n=0;n<ngblist[label-1].length;n++) {
                     if (ngblb==ngblist[label-1][n]) {
                         loc=n;
                         n=ngblist[label-1].length;
                     }
                }
                if (loc>-1 && bdcount[label-1][loc]>mincount && noise[label-1]>0.0 && noise[ngblb-1]>0.0) {
                    double sigmaAB = (incount[label-1]*noise[label-1]*noise[label-1] + incount[ngblb-1]*noise[ngblb-1]*noise[ngblb-1]
                                        + incount[label-1]*incount[ngblb-1]/(incount[label-1]+incount[ngblb-1])
                                            *(interior[label-1]-interior[ngblb-1])*(interior[label-1]-interior[ngblb-1]))
                                                /(incount[label-1]+incount[ngblb-1]);
                                                
                    double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                    - 0.5*incount[label-1]/Numerics.max(1.0,(incount[label-1]+incount[ngblb-1]))
                                        *FastMath.log(noise[label-1]*noise[label-1])
                                    - 0.5*incount[ngblb-1]/Numerics.max(1.0,(incount[label-1]+incount[ngblb-1]))
                                        *FastMath.log(noise[ngblb-1]*noise[ngblb-1]);
                    boundaries[xyz] = (float)FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0);
                }
            }
        }
                
	    return;
	}
	
	// multi-region growing of supervoxels from high probabilities to lower ones with likely transition
	public void computeParcelSegmentation() {
	    // we assume we have the distance to the boundary and the label of the closest supervoxel
	    // now we fit a sigmoid to intensity, distance for each boundary
	    // the sigmoid has three parameters: height (contrast), slope (sharpness) and offset (true boundary)
	    
	    // we assume a probabilistic prior separation into 3 regions, with the boundary being centered
	    // at zero and the slope being two voxels in spread
	    // thus inside is 1-exp(-|lvl|/d0), and boundary is exp(-|lvl|/d0)
	    
	    float[] interior = new float[nsxyz];
	    float[] incount = new float[nsxyz];
	    
	    float delta = 0.001f;
	    float delta0 = 0.1f;
	    float dist0 = 2.0f;
	    //int nngb = 30;
	    
	    // use HashSets to count the number of needed neighbors per location
	    HashSet[] ngbset = new HashSet[nsxyz];
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        ngbset[xyzs] = new HashSet(nngb);
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        ngbset[label-1].add(ngblb);
	    }
	    // count the number of neighbors per region??
	    // must build a full list...
	    int[][] ngblist = new int[nsxyz][];
	    float[][] bdproba = new float[nsxyz][];	    
	    
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	         ngblist[xyzs] = new int[ngbset[xyzs].size()];
	         bdproba[xyzs] = new float[ngbset[xyzs].size()];
	    }
	    
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        boolean found=false;
	        int last = ngblist[label-1].length;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                found=true;
	                n=ngblist[label-1].length;
	            } else if (ngblist[label-1][n]==0) {
	                ngblist[label-1][n] = ngblb;
	                n=ngblist[label-1].length;
	            }
	        }
	    }
	    
	    // boundaries: use all the interior of a given region
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        interior[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*inputImage[xyz];
	        incount[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)));
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) interior[xyzs] /= incount[xyzs];
	    }
	    	    
	    // boundary SNR
	    float[] noise = new float[nsxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        noise[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*Numerics.square(inputImage[xyz]-interior[label-1]);
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) noise[xyzs] = (float)FastMath.sqrt(noise[xyzs]/incount[xyzs]);
	    }
        // boundary probability based on Jensen-Shannon divergence
        for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) {
            int label =xyzs+1;
            int ngblb = ngblist[xyzs][ngb];
                
            if (noise[label-1]>0.0 && noise[ngblb-1]>0.0 && incount[label-1]>0 && incount[ngblb-1]>0) {
                double sigmaAB = (incount[label-1]*noise[label-1]*noise[label-1] + incount[ngblb-1]*noise[ngblb-1]*noise[ngblb-1]
                                    + incount[label-1]*incount[ngblb-1]/(incount[label-1]+incount[ngblb-1])
                                        *(interior[label-1]-interior[ngblb-1])*(interior[label-1]-interior[ngblb-1]))
                                            /(incount[label-1]+incount[ngblb-1]);
                if (sigmaAB>0.0) {                                
                    double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                    - 0.5*incount[label-1]/(incount[label-1]+incount[ngblb-1])
                                        *FastMath.log(noise[label-1]*noise[label-1])
                                    - 0.5*incount[ngblb-1]/(incount[label-1]+incount[ngblb-1])
                                        *FastMath.log(noise[ngblb-1]*noise[ngblb-1]);
                    bdproba[xyzs][ngb] = (float)Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                }
            }
        }
        // use mean, stdev to define a sigmoid transform (?)
        double avg = 0.0;
        double std = 0.0;
        double den = 0.0;
        for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) if (bdproba[xyzs][ngb]>0) {
            avg += bdproba[xyzs][ngb];
            den++;
        }
        if (den>0) avg /= den;
        for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) if (bdproba[xyzs][ngb]>0) {
            std += (bdproba[xyzs][ngb]-avg)*(bdproba[xyzs][ngb]-avg);
        }
        if (den>0) std /= den;
        std = FastMath.sqrt(std);
        
        System.out.println("Boundaries distribution (m,s): "+avg+", "+std);
        for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) if (bdproba[xyzs][ngb]>0) {
            bdproba[xyzs][ngb] = (float)(1.0/(1.0+FastMath.exp(-(bdproba[xyzs][ngb]-avg)/(std))));
        }
        
        // need the number of possible classes
        if (initsegImage==null) {
            // build a segmentation iamge from priors
            initsegImage = new int[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                int best=0;
                for (int n=1;n<nprior;n++) if (priorImage[n][xyz]>priorImage[best][xyz]) best=n;
                if (priorImage[best][xyz]>0) initsegImage[xyz] = best+1;
            }    
        }
        int[] lbseg = ObjectLabeling.listOrderedNonzeroLabels(initsegImage, nx, ny, nz);
        int nseg = lbseg.length;
        
        /* not useful
        // global stats for intensity priors
        float[] meanseg = new float[nseg];
        float[] sumseg = new float[nseg];
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
            int seg=-1;
	        for (int n=0;n<nseg;n++) if (lbseg[n]==initsegImage[xyz]) {
	            seg=n;
	            n=nseg;
	        }
	        if (seg>-1) {
	            meanseg[seg] += inputImage[xyz];
	            sumseg[seg]++;
	        }
        }
        for (int n=0;n<nseg;n++) if (sumseg[n]>0) meanseg[n] /= sumseg[n];

        float[] varseg = new float[nseg];
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
            int seg=-1;
	        for (int n=0;n<nseg;n++) if (lbseg[n]==initsegImage[xyz]) {
	            seg=n;
	            n=nseg;
	        }
	        if (seg>-1) {
	            varseg[seg] += (inputImage[xyz]-meanseg[seg])*(inputImage[xyz]-meanseg[seg]);
	        }
        }
        for (int n=0;n<nseg;n++) if (sumseg[n]>0) varseg[n] /= sumseg[n];
        
        float varmax = 0.0f;
        for (int n=0;n<nseg;n++) {
        //    System.out.println("class "+lbseg[n]+": "+meanseg[n]+" +/- "+FastMath.sqrt(varseg[n]));
            if (varseg[n]>varmax) varmax = varseg[n];
        }*/
        
	    float[][] parcelmem = new float[nsxyz][nseg];
	    //int[] parcelseg = new int[nsxyz];
        
        // for each parcel (not voxels), find best candidate and propagate priors from neighbors
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        int seg=-1;
	        for (int n=0;n<nseg;n++) if (lbseg[n]==initsegImage[xyz]) {
	            seg=n;
	            n=nseg;
	        }
	        if (seg>-1) {
	            // note: here we use the same weighting function that makes border regions less impactful, maybe not wise?
	            if (nprior==1)
                    parcelmem[label-1][seg] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*priorImage[0][xyz];
                else
                    parcelmem[label-1][seg] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*priorImage[seg][xyz];
            }
	    }
	    initsegImage = null;
	    priorImage = null;
	    
	    // average over ROI
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) {
	            for (int n=0;n<nseg;n++)  {
	                parcelmem[xyzs][n] /= incount[xyzs];
	            }
	        }
	    }
	    /*
	    // normalize to 1: what is important here is the homogeneity of the parcellations?
	    // seems to work marginally better without now
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        float sum=0.0f;
	        for (int n=0;n<nseg;n++) sum += parcelmem[xyzs][n];
	        if (sum>0) {
	            for (int n=0;n<nseg;n++) parcelmem[xyzs][n] /= sum;
	        }
	    }*/
	    /* not needed
	    // intensity average per supervoxel, same calculation
	    float[] parcelimg = new float[nsxyz];
	    
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        parcelimg[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*inputImage[xyz];
 	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) parcelimg[xyzs] /= incount[xyzs];
	    }
	    */
	             
	    // run the propagation across neighbors (iterative)
	    float wngb = 0.5f;
	    float[][] newmem = new float[nsxyz][nseg];
	    float diff=maxdiff;
        for (int t=0;t<maxiter && diff>=maxdiff;t++) {
            for (int xyzs=0;xyzs<nsxyz;xyzs++) {
                if (t==0) for (int n=0;n<nseg;n++) {
                    newmem[xyzs][n] = parcelmem[xyzs][n];
                }
                // for all neighbors and classes, update if neighbor higher/lower 
                //double[] count = new double[nseg];
                for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) if (bdproba[xyzs][ngb]>0) {
                    // check only the most likely neighbor
                    float best=0.0f;
                    int nbest=-1;
                    for (int n=0;n<nseg;n++) {
                        if (parcelmem[ngblist[xyzs][ngb]-1][n]>best) {
                            best = parcelmem[ngblist[xyzs][ngb]-1][n];
                            nbest = n;
                        }
                    }
                    if (nbest>-1) {
                        double count = 1.0;
                        double ngbsame = FastMath.sqrt((1.0-bdproba[xyzs][ngb])*parcelmem[ngblist[xyzs][ngb]-1][nbest]);
                        if (ngbsame>parcelmem[xyzs][nbest]) {
                            newmem[xyzs][nbest] += (float)(ngbsame);
                            count += 1.0;
                        }
                        double ngbdiff = FastMath.sqrt(bdproba[xyzs][ngb]*parcelmem[ngblist[xyzs][ngb]-1][nbest]);
                        if (ngbdiff>parcelmem[xyzs][nbest]) {
                            newmem[xyzs][nbest] += (float)(parcelmem[xyzs][nbest]*parcelmem[xyzs][nbest]/ngbdiff);
                            count += 1.0;
                        }
                        newmem[xyzs][nbest] /= (float)count;
                    }
                }
                //for (int n=0;n<nseg;n++) newmem[xyzs][n] /= (float)(1.0+count[n]);
                // renormalize to 1? lower the overall probabilities 
                /*
                float norm=0.0f;
                for (int n=0;n<nseg;n++) norm += newmem[xyzs][n];
                if (norm>0) for (int n=0;n<nseg;n++) newmem[xyzs][n] /= norm;
                */
	        }
            diff = 0.0f;
            for (int xyzs=0;xyzs<nsxyz;xyzs++) {
                for (int n=0;n<nseg;n++) {
                    diff = Numerics.max(diff,Numerics.abs(parcelmem[xyzs][n]-newmem[xyzs][n]));
                    
                    parcelmem[xyzs][n] = newmem[xyzs][n];    
                }
            }
            System.out.println("iter "+t+", max diff: "+diff);
        }
            
	    // export the result
	    segImage = new int[nxyz];
	    memsImage = new float[nxyz];
	    /*
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        int best=0;
	        for (int n=1;n<nseg;n++) if (parcelmem[label-1][n]>parcelmem[label-1][best]) best=n;
	        
	        segImage[xyz] = lbseg[best];
	        memsImage[xyz] = parcelmem[label-1][best];
	    }*/
	    // build a new image from flat region intensities (use original estimates, not final ones)
	    /*
	    float[] inparcel = new float[nsxyz];
	    float[] inseg = new float[nseg];
	    float[] sumparcel = new float[nsxyz];
	    float[] sumseg = new float[nseg];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        int best=0;
	        for (int n=1;n<nseg;n++) if (parcelmem[label-1][n]>parcelmem[label-1][best]) best=n;
	        
	        segImage[xyz] = best;
	        
	        inparcel[label-1] += inputImage[xyz];
	        sumparcel[label-1]++;
	        
	        inseg[best] += inputImage[xyz];
	        sumseg[best]++;
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) if (sumparcel[xyzs]>0) inparcel[xyzs] /= sumparcel[xyzs];
	    for (int n=0;n<nseg;n++) if (sumseg[n]>0) inseg[n] /= sumseg[n];
	    */
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        int best=0;
	        for (int n=1;n<nseg;n++) if (parcelmem[label-1][n]>parcelmem[label-1][best]) best=n;

	        segImage[xyz] = lbseg[best];
	        memsImage[xyz] = parcelmem[label-1][best];
	        //memsImage[xyz] = inputImage[xyz] - inparcel[label-1] + inseg[best];
	        //memsImage[xyz] = inputImage[xyz] - parcelimg[label-1] + meanseg[best];
	    }
	    return;
	}
	
	public void computeParcelAggregation(float maxproba) {
	    // we assume we have the distance to the boundary and the label of the closest supervoxel
	    // now we fit a sigmoid to intensity, distance for each boundary
	    // the sigmoid has three parameters: height (contrast), slope (sharpness) and offset (true boundary)
	    
	    // we assume a probabilistic prior separation into 3 regions, with the boundary being centered
	    // at zero and the slope being two voxels in spread
	    // thus inside is 1-exp(-|lvl|/d0), and boundary is exp(-|lvl|/d0)
	    
	    float[] interior = new float[nsxyz];
	    float[] incount = new float[nsxyz];
	    
	    float delta = 0.001f;
	    float delta0 = 0.1f;
	    float dist0 = 2.0f;
	    //int nngb = 30;
	    
	    // use HashSets to count the number of needed neighbors per location
	    HashSet[] ngbset = new HashSet[nsxyz];
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        ngbset[xyzs] = new HashSet(nngb);
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        ngbset[label-1].add(ngblb);
	    }
	    // count the number of neighbors per region??
	    // must build a full list...
	    int[][] ngblist = new int[nsxyz][];
	    float[][] bdproba = new float[nsxyz][];	    
	    
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	         ngblist[xyzs] = new int[ngbset[xyzs].size()];
	         bdproba[xyzs] = new float[ngbset[xyzs].size()];
	    }
	    
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcel[xyz];
	        int ngblb = neighbor[xyz];
	        
	        boolean found=false;
	        int last = ngblist[label-1].length;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                found=true;
	                n=ngblist[label-1].length;
	            } else if (ngblist[label-1][n]==0) {
	                ngblist[label-1][n] = ngblb;
	                n=ngblist[label-1].length;
	            }
	        }
	    }
	    
	    // boundaries: use all the interior of a given region
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        interior[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*inputImage[xyz];
	        incount[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)));
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) interior[xyzs] /= incount[xyzs];
	    }
	    	    
	    // boundary SNR
	    float[] noise = new float[nsxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        noise[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*Numerics.square(inputImage[xyz]-interior[label-1]);
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) noise[xyzs] = (float)FastMath.sqrt(noise[xyzs]/incount[xyzs]);
	    }
        // boundary probability based on Jensen-Shannon divergence
        for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) {
            int label = xyzs+1;
            int ngblb = ngblist[xyzs][ngb];
            
            // only map the boundary once
            if (ngblb>label) {
                if (noise[label-1]>0.0 && noise[ngblb-1]>0.0 && incount[label-1]>0 && incount[ngblb-1]>0) {
                    double sigmaAB = (incount[label-1]*noise[label-1]*noise[label-1] + incount[ngblb-1]*noise[ngblb-1]*noise[ngblb-1]
                                        + incount[label-1]*incount[ngblb-1]/(incount[label-1]+incount[ngblb-1])
                                            *(interior[label-1]-interior[ngblb-1])*(interior[label-1]-interior[ngblb-1]))
                                                /(incount[label-1]+incount[ngblb-1]);
                    if (sigmaAB>0.0) {                                
                        double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                        - 0.5*incount[label-1]/(incount[label-1]+incount[ngblb-1])
                                            *FastMath.log(noise[label-1]*noise[label-1])
                                        - 0.5*incount[ngblb-1]/(incount[label-1]+incount[ngblb-1])
                                            *FastMath.log(noise[ngblb-1]*noise[ngblb-1]);
                        bdproba[xyzs][ngb] = (float)Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                    }
                }
            }
        }
        // from there onward, do aggregative clustering
        int[] cluster = new int[nsxyz];        
        float[] clsum = new float[nsxyz];
        float[] clvar = new float[nsxyz];
        float[] clmin = new float[nsxyz];
        float[] clmax = new float[nsxyz];
        
        // 1. Make a (large) array with all boundaries below maxproba
        
        // computation variables
        BinaryHeapPair heap = new BinaryHeapPair(nsx*nsy+nsy*nsz+nsz*nsx, BinaryHeapPair.MINTREE);
		heap.reset();
        for (int xyzs=0;xyzs<nsxyz;xyzs++) {
            for (int ngb=0;ngb<ngblist[xyzs].length;ngb++) if (bdproba[xyzs][ngb]>0) {
                // add to the heap with boundary value
                if (bdproba[xyzs][ngb]<maxproba) heap.addValue(bdproba[xyzs][ngb],xyzs,ngb);
            }
            // init local quantities
            clsum[xyzs] = incount[xyzs];
            clmin[xyzs] = interior[xyzs];
            clmax[xyzs] = interior[xyzs];
            clvar[xyzs] = noise[xyzs]*noise[xyzs];
        }
		// grow the labels and functions
        float maxdist = 0.0f;
        int maxcluster = 0;
        while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int xyzs = heap.getFirstId1();
        	int ngb = heap.getFirstId2();
			heap.removeFirst();
			
			int lb1 = xyzs+1;
			int lb2 = ngblist[xyzs][ngb];

            // check if the probability has changed
            if (bdproba[xyzs][ngb]>dist) {
                if (bdproba[xyzs][ngb]<maxproba) heap.addValue(bdproba[xyzs][ngb],xyzs,ngb);
                continue;
            } else if (bdproba[xyzs][ngb]<dist) {
                // skip if already lower and assigned
                continue;
            }
            // same value: update
            
            // proceed case by case for the whole thing, more efficient...
			if (cluster[lb1-1]==0 && cluster[lb2-1]==0) {
			    // both new

			    // update stats
			    clvar[lb1-1] = Numerics.min(clvar[lb1-1],clvar[lb2-1]);
			    clsum[lb1-1] = Numerics.max(clsum[lb1-1],clsum[lb2-1]);
			    clmin[lb1-1] = Numerics.min(clmin[lb1-1],clmin[lb2-1]);
                clmax[lb1-1] = Numerics.max(clmax[lb1-1],clmax[lb2-1]);
                
			    clvar[lb2-1] = clvar[lb1-1];
			    clsum[lb2-1] = clsum[lb1-1];
			    clmin[lb2-1] = clmin[lb1-1];
			    clmax[lb2-1] = clmax[lb1-1];
			    
			    // update labels
                maxcluster++;
			    cluster[lb1-1] = maxcluster;
			    cluster[lb2-1] = maxcluster;
			    
                // update local neighbors with new values
			    for (int nb=0;nb<ngblist[lb1-1].length;nb++) if (bdproba[lb1-1][nb]>0) {
			        int label = lb2;
			        int ngblb = ngblist[lb1-1][nb];
			        if (cluster[ngblb-1]!=cluster[lb1-1]) {
                
                        double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                        double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                        - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[label-1])
                                        - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[ngblb-1]);
                        double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                        
                        // always update
                        if (score<bdproba[lb1-1][nb] && score<maxproba) heap.addValue((float)score,lb1-1,nb);
                        bdproba[lb1-1][nb] = (float)score;
                    }
                }
			    for (int nb=0;nb<ngblist[lb2-1].length;nb++) if (bdproba[lb2-1][nb]>0) {
			        int label = lb1;
			        int ngblb = ngblist[lb2-1][nb];
			        if (cluster[ngblb-1]!=cluster[lb2-1]) {
                
                        double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                        double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                        - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[label-1])
                                        - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[ngblb-1]);
                        double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                        
                        // always update
                        if (score<bdproba[lb2-1][nb] && score<maxproba) heap.addValue((float)score,lb2-1,nb);
                        bdproba[lb2-1][nb] = (float)score;
                    }
                }
			    
			    System.out.print("+");
            } else if (cluster[lb1-1]==0 && cluster[lb2-1]>0) {
			    // first new, second already a cluster
			    
                // update stats
			    clvar[lb1-1] = Numerics.min(clvar[lb1-1],clvar[lb2-1]);
			    clsum[lb1-1] = Numerics.max(clsum[lb1-1],clsum[lb2-1]);
			    clmin[lb1-1] = Numerics.min(clmin[lb1-1],clmin[lb2-1]);
                clmax[lb1-1] = Numerics.max(clmax[lb1-1],clmax[lb2-1]);
                
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb2-1]) {
                    int lbn = loc+1;
                    clvar[lbn-1] = clvar[lb1-1];
                    clsum[lbn-1] = clsum[lb1-1];
                    clmin[lbn-1] = clmin[lb1-1];
                    clmax[lbn-1] = clmax[lb1-1];
			    }

			    // update labels
			    cluster[lb1-1] = cluster[lb2-1];

			    // update local neighbors
			    for (int nb=0;nb<ngblist[lb1-1].length;nb++) if (bdproba[lb1-1][nb]>0) {
			        int label = lb2;
			        int ngblb = ngblist[lb1-1][nb];
			        
			        if (cluster[ngblb-1]!=cluster[lb1-1]) {
                        double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                        double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                        - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[label-1])
                                        - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[ngblb-1]);
                        double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                    
                        // always update
                        if (score<bdproba[lb1-1][nb] && score<maxproba) heap.addValue((float)score,lb1-1,nb);
                        bdproba[lb1-1][nb] = (float)score;
                    }
                }
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb2-1]) {
			        int lbn = loc+1;
                    for (int nb=0;nb<ngblist[lbn-1].length;nb++) if (bdproba[lbn-1][nb]>0 && cluster[ngblist[lbn-1][nb]-1]!=cluster[lb2-1]) {
                        int label = lb1;
                        int ngblb = ngblist[lbn-1][nb];
                        
                        if (cluster[ngblb-1]!=cluster[lbn-1]) {
                            double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                            + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                           double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                            - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[label-1])
                                            - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[ngblb-1]);
                            double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                            
                            // always update
                            if (score<bdproba[lbn-1][nb] && score<maxproba) heap.addValue((float)score,lbn-1,nb);
                            bdproba[lbn-1][nb] = (float)score;
                        }
                    }
                }

			    System.out.print("1");
			} else if (cluster[lb1-1]>0 && cluster[lb2-1]==0) {
			    // second new, first already a cluster
			    
                // update stats
			    clvar[lb2-1] = Numerics.min(clvar[lb1-1],clvar[lb2-1]);
			    clsum[lb2-1] = Numerics.max(clsum[lb1-1],clsum[lb2-1]);
			    clmin[lb2-1] = Numerics.min(clmin[lb1-1],clmin[lb2-1]);
                clmax[lb2-1] = Numerics.max(clmax[lb1-1],clmax[lb2-1]);
                
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb1-1]) {
                    int lbn = loc+1;
                    clvar[lbn-1] = clvar[lb2-1];
                    clsum[lbn-1] = clsum[lb2-1];
                    clmin[lbn-1] = clmin[lb2-1];
                    clmax[lbn-1] = clmax[lb2-1];
			    }

                // update labels
			    cluster[lb2-1] = cluster[lb1-1];

			    // update local neighbors
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb1-1]) {
			        int lbn = loc+1;
			        for (int nb=0;nb<ngblist[lbn-1].length;nb++) if (bdproba[lbn-1][nb]>0 && cluster[ngblist[lbn-1][nb]-1]!=cluster[lb1-1]) {
			            int label = lb2;
			            int ngblb = ngblist[lbn-1][nb];
                
                        if (cluster[ngblb-1]!=cluster[lbn-1]) {
                            double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                            + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                            double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                            - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[label-1])
                                            - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[ngblb-1]);
                            double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                            
                            // always update
                            if (score<bdproba[lbn-1][nb] && score<maxproba) heap.addValue((float)score,lbn-1,nb);
                            bdproba[lbn-1][nb] = (float)score;
                        }
                    }
                }
			    for (int nb=0;nb<ngblist[lb2-1].length;nb++) if (bdproba[lb2-1][nb]>0) {
			        int label = lb1;
			        int ngblb = ngblist[lb2-1][nb];
			                        
			        if (cluster[ngblb-1]!=cluster[lb2-1]) {
                        double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                        double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                        - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[label-1])
                                        - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *FastMath.log(clvar[ngblb-1]);
                        double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                    
                        // always update
                        if (score<bdproba[lb2-1][nb] && score<maxproba) heap.addValue((float)score,lb2-1,nb);
                        bdproba[lb2-1][nb] = (float)score;
                    }
                }

			    System.out.print("2");
			} else if (cluster[lb1-1]>0 && cluster[lb2-1]>0) {
			    // none new, both clusters
			    
                // update stats
			    clvar[lb1-1] = Numerics.min(clvar[lb1-1],clvar[lb2-1]);
			    clsum[lb1-1] = Numerics.max(clsum[lb1-1],clsum[lb2-1]);
			    clmin[lb1-1] = Numerics.min(clmin[lb1-1],clmin[lb2-1]);
                clmax[lb1-1] = Numerics.max(clmax[lb1-1],clmax[lb2-1]);
                
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb1-1]) {
                    int lbn = loc+1;
                    clvar[lbn-1] = clvar[lb1-1];
                    clsum[lbn-1] = clsum[lb1-1];
                    clmin[lbn-1] = clmin[lb1-1];
                    clmax[lbn-1] = clmax[lb1-1];
			    }
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb2-1]) {
                    int lbn = loc+1;
                    clvar[lbn-1] = clvar[lb1-1];
                    clsum[lbn-1] = clsum[lb1-1];
                    clmin[lbn-1] = clmin[lb1-1];
                    clmax[lbn-1] = clmax[lb1-1];
			    }
                
			    // update labels
                if (cluster[lb1-1]<cluster[lb2-1]) {
                    for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb2-1]) {
			            int lbn = loc+1;
                        cluster[lbn-1] = cluster[lb1-1];
                    }
                } else {
                    for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb1-1]) {
			            int lbn = loc+1;
                        cluster[lbn-1] = cluster[lb2-1];
                    }
                }

			    // update local neighbors
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb1-1]) {
			        int lbn = loc+1;
			        for (int nb=0;nb<ngblist[lbn-1].length;nb++) if (bdproba[lbn-1][nb]>0 && cluster[ngblist[lbn-1][nb]-1]!=cluster[lb1-1]) {
			            int label = lb2;
			            int ngblb = ngblist[lbn-1][nb];
                
                        if (cluster[ngblb-1]!=cluster[lbn-1]) {
                            double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                            double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                            - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[label-1])
                                            - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[ngblb-1]);
                            double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                            
                            // always update
                            if (score<bdproba[lbn-1][nb] && score<maxproba) heap.addValue((float)score,lbn-1,nb);
                            bdproba[lbn-1][nb] = (float)score;
                        }
                    }
                }
                for (int loc=0;loc<nsxyz;loc++) if (cluster[loc]==cluster[lb2-1]) {
			        int lbn = loc+1;
			        for (int nb=0;nb<ngblist[lbn-1].length;nb++) if (bdproba[lbn-1][nb]>0 && cluster[ngblist[lbn-1][nb]-1]!=cluster[lb2-1]) {
			            int label = lb1;
			            int ngblb = ngblist[lbn-1][nb];
                
                        if (cluster[ngblb-1]!=cluster[lbn-1]) {
                            double sigmaAB = (clsum[label-1]*clvar[label-1] + clsum[ngblb-1]*clvar[ngblb-1]
                                        + clsum[label-1]*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                            *Numerics.max((clmin[label-1]-clmax[ngblb-1])*(clmin[label-1]-clmax[ngblb-1]),
                                                          (clmax[label-1]-clmin[ngblb-1])*(clmax[label-1]-clmin[ngblb-1])))
                                                /(clsum[label-1]+clsum[ngblb-1]);
                            double jsdiv = 0.5*FastMath.log(sigmaAB) 
                                            - 0.5*clsum[label-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[label-1])
                                            - 0.5*clsum[ngblb-1]/(clsum[label-1]+clsum[ngblb-1])
                                                *FastMath.log(clvar[ngblb-1]);
                            double score = Numerics.min(1.0,FastMath.sqrt(jsdiv*FastMath.log(2.0)/2.0));
                            
                            // always update
                            if (score<bdproba[lbn-1][nb] && score<maxproba) heap.addValue((float)score,lbn-1,nb);
                            bdproba[lbn-1][nb] = (float)score;
                        }
                    }
                }
                
			    System.out.print("-");
			}
			
		}
		
		float[] meanc = new float[maxcluster];
		float[] sumc = new float[maxcluster];
		for (int xyzs=0;xyzs<nsxyz;xyzs++) if (cluster[xyzs]>0) {
		    meanc[cluster[xyzs]-1] += rescaled[xyzs];
		    sumc[cluster[xyzs]-1]++;
		}
		for (int c=0;c<maxcluster;c++) if (sumc[c]>0) meanc[c] /= sumc[c];
		
		// output cluster map
	    segImage = new int[nxyz];
	    memsImage = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        int label = parcel[xyz];
	        
	        segImage[xyz] = cluster[label-1];
	        if (cluster[label-1]>0) {
	            memsImage[xyz] = meanc[cluster[label-1]-1];
	        } else {
	            memsImage[xyz] = rescaled[label-1];
	        }
	    }
	    
	    // TODO: use the clustering to update the input segmentation
	    
	    
	    return;
	}

}
