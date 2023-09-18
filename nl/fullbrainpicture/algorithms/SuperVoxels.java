package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.HashSet;

/*
 * @author Pierre-Louis Bazin
 */
public class SuperVoxels {

	// jist containers
	private float[] inputImage=null;
	private int[] maskImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int nsx, nsy, nsz, nsxyz;
	
	private float scaling;
	private float noise;
	private int nngb = 26;

	private int[] parcelImage;
	private float[] rescaledImage;
	private float[] memsImage;
	
	// intermediate results
	private int[] count;
	private boolean[] mask;
	private float[] levelset;
	private int[] neighbor;
	
	private String outputs = "average";
	
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

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	public final void setScalingFactor(float val) { scaling = val; }
	public final void setNoiseLevel(float val) { noise = val; }
	
	public final void setOutputType(String val) { outputs = val; }
	
	// create outputs
	public final int[] getParcelImage() { return parcelImage; }
	public final float[] getRescaledImage() { return rescaledImage; }
	public final float[] getMemsImage() { return memsImage; }
	
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
	    
	    // Compute the supervoxel grid
	    System.out.println("original dimensions: ("+nx+", "+ny+", "+nz+")");
	    nsx = Numerics.floor(nx/scaling);
	    nsy = Numerics.floor(ny/scaling);
	    nsz = Numerics.floor(nz/scaling);
	    nsxyz = nsx*nsy*nsz;
	    System.out.println("rescaled dimensions: ("+nsx+", "+nsy+", "+nsz+")");
	    
	    // init downscaled images
	    parcelImage = new int[nxyz];
	    rescaledImage = new float[nsxyz];
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
	            parcelImage[xyz] = xyzs+1;
	            rescaledImage[xyzs] = inputImage[xyz];
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
                                    
                            float contrast = (inputImage[xyznb]-rescaledImage[xyzs])
                                            *(inputImage[xyznb]-rescaledImage[xyzs]);
                                        
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
			parcelImage[xyz] = xyzs+1;
            rescaledImage[xyzs] = count[xyzs]*rescaledImage[xyzs] + inputImage[xyz];
            
            centroid[X][xyzs] = count[xyzs]*centroid[X][xyzs] + x;
	        centroid[Y][xyzs] = count[xyzs]*centroid[Y][xyzs] + y;
	        centroid[Z][xyzs] = count[xyzs]*centroid[Z][xyzs] + z;
	        
	        count[xyzs] += 1;
	        rescaledImage[xyzs] /= count[xyzs];
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
                                
                        float contrast = (inputImage[xyznb]-rescaledImage[xyzs])
                                        *(inputImage[xyznb]-rescaledImage[xyzs]);
                                        
                        heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                    }
                }
            }
		}
		// Output informations
		if (outputs.equals("average")) {
		    memsImage = new float[nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) {
		        if (parcelImage[xyz]>0) {
		            memsImage[xyz] = rescaledImage[parcelImage[xyz]-1];
		        }
		    }
		} else if (outputs.equals("difference")) {
		    memsImage = new float[nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) {
		        if (parcelImage[xyz]>0) {
		            memsImage[xyz] = Numerics.abs(inputImage[xyz]-rescaledImage[parcelImage[xyz]-1]);
		        }
		    }
		} else if (outputs.equals("distance")) {
		    growBoundaries();
		    memsImage = new float[nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) {
		        if (parcelImage[xyz]>0) {
		            memsImage[xyz] = levelset[xyz];
		        }
		    }
		} else if (outputs.equals("neighbor")) {
		    growBoundaries();
		    memsImage = new float[nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) {
		        if (parcelImage[xyz]>0) {
		            memsImage[xyz] = neighbor[xyz];
		        }
		    }
		} else {
		    System.out.println("Estimate boundary sharpness");
		    growBoundaries();
            fitBoundarySigmoid();
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
                    if (cmask[xyzn] && parcelImage[xyz]>0 && parcelImage[xyzn]>0 && parcelImage[xyzn]!=parcelImage[xyz]) {
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
			neighbor[xyz] = parcelImage[ngb];
			processed[xyz]=true; // update the current level
 			
			// find new neighbors
			for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				
				// must be in outside the object or its processed neighborhood
				if (cmask[xyzn] && !processed[xyzn]) if (parcelImage[xyzn]==parcelImage[xyz]) {
					// compute new distance based on processed neighbors for the same object
					for (byte l=0; l<6; l++) {
						nbdist[l] = -1.0f;
						nbflag[l] = false;
						int xyznb = ObjectTransforms.fastMarchingNeighborIndex(l, xyzn, nx, ny, nz);
						// note that there is at most one value used here
						if (cmask[xyznb] && processed[xyznb]) if (parcelImage[xyznb]==parcelImage[xyz]) {
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
	
	public void fitBoundarySigmoid() {
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
	    
	    // boundaries: use all the interior of a given region
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        
	        interior[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*inputImage[xyz];
	        incount[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)));
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) interior[xyzs] /= incount[xyzs];
	    }
	    
	    // boundary SNR
	    float[] noise = new float[nsxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        
	        noise[label-1] += (float)(1.0-FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)))*Numerics.square(inputImage[xyz]-interior[label-1]);
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) noise[xyzs] = (float)FastMath.sqrt(noise[xyzs]/incount[xyzs]);
	    }
	    
	    // use HashSets to count the number of needed neighbors per location
	    HashSet[] ngbset = new HashSet[nsxyz];
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        ngbset[xyzs] = new HashSet(nngb);
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	        
	        ngbset[label-1].add(ngblb);
	    }
	    // count the number of neighbors per region??
	    // must build a full list...
	    int[][] ngblist = new int[nsxyz][];
	    int[][] bdcount = new int[nsxyz][];
	    float[][] slope = new float[nsxyz][];
	    float[][] dist = new float[nsxyz][];

	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	         ngblist[xyzs] = new int[ngbset[xyzs].size()];
	         bdcount[xyzs] = new int[ngbset[xyzs].size()];
	         slope[xyzs] = new float[ngbset[xyzs].size()];
	         dist[xyzs] = new float[ngbset[xyzs].size()];
	    }
	    
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	        
	        boolean found=false;
	        int last = ngblist[label-1].length;
	        for (int n=0;n<ngblist[label-1].length;n++) {
	            if (ngblb==ngblist[label-1][n]) {
	                found=true;
	                bdcount[label-1][n]++;
	                n=ngblist[label-1].length;
	            } else if (ngblist[label-1][n]==0) {
	                last=n;
	                n=ngblist[label-1].length;
	            }
	        }
	        if (!found && last<nngb) {
	            ngblist[label-1][last] = ngblb;
	            bdcount[label-1][last] = 1;
	        }
	    }
	    
	    
	    // offset not recomputed for simplicity, but boundary count is still relevant?
	    // maybe count directly rather
	    /*
	    float[][] offset = new float[nngb][nsxyz];
	    float[][] bdcount = new float[nngb][nsxyz];
	    
	    // offset
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	        
            float wgty = Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta)
                        *Numerics.bounded((interior[label-1]-inputImage[xyz])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta);
            float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)), delta, 1.0f-delta);
                
	        // find the neighbor in the list
	        int loc = -1;
	        for (int n=0;n<nngb;n++) {
	             if (ngblb==ngblist[n][label-1]) {
	                 loc=n;
	                 n=nngb;
	             }
	        }
	        if (loc>-1) {
                offset[loc][label-1] += levelset[xyz]*wgtx*wgty;
                bdcount[loc][label-1] += wgtx*wgty;    
            }
            
            // also in the neighbor's for negative values
            loc = -1;
	        for (int n=0;n<nngb;n++) {
	             if (label==ngblist[n][ngblb-1]) {
	                 loc=n;
	                 n=nngb;
	             }
	        }
	        if (loc>-1) {
                offset[loc][ngblb-1] += -levelset[xyz]*wgtx*wgty;
                bdcount[loc][ngblb-1] += wgtx*wgty;    
            }
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<nngb;c++) {
	        if (bdcount[c][xyzs]>0) offset[c][xyzs] /= bdcount[c][xyzs];
	    }
	    */
	    
	    // slope
	    for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
	        int label = parcelImage[xyz];
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
	        if (loc>-1) {
	            slope[label-1][loc] += (2.0f*Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta)-1.0f)
                                            *wgtx*wgty;
                    
                //dist[loc][label-1] += (levelset[xyz]-offset[loc][label-1])*wgtx*wgty;
                dist[label-1][loc] += levelset[xyz]*wgtx*wgty;
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
                slope[ngblb-1][loc] += -(2.0f*Numerics.bounded((inputImage[xyz]-interior[label-1])/(interior[ngblb-1]-interior[label-1]), delta, 1.0f-delta)-1.0f)
                                            *wgtx*wgty;
                                            
                //dist[loc][ngblb-1] += -(-levelset[xyz]-offset[loc][ngblb-1])*wgtx*wgty;
                dist[ngblb-1][loc] += levelset[xyz]*wgtx*wgty;
            }
            
        }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<ngblist[xyzs].length;c++) {
	        if (dist[xyzs][c]>0) slope[xyzs][c] = 0.5f*slope[xyzs][c]/dist[xyzs][c];
	    }
	        
	    // outputs?
	    System.out.println("Output map: "+outputs);
	    memsImage = new float[nxyz];
	    if (outputs.equals("boundary")) {	    
	        for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
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
                    memsImage[xyz] = loc;
                }
            }
        } else
	    if (outputs.equals("baseline")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                memsImage[xyz] = interior[label-1];
            }
        } else
	    if (outputs.equals("count")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
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
                    if (loc>-1) 
                        memsImage[xyz] = bdcount[label-1][loc];
                }
            }
        } else
	    if (outputs.equals("contrast")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                if (levelset[xyz]<1.0f) {
                    memsImage[xyz] = Numerics.abs(interior[label-1]-interior[ngblb-1]);
                }
            }
        } else
	    if (outputs.equals("noise")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                memsImage[xyz] = noise[label-1];
            }
        } else
	    if (outputs.equals("CNR")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                if (levelset[xyz]<1.0f) {
                    memsImage[xyz] = 2.0f*Numerics.abs(interior[label-1]-interior[ngblb-1])/Numerics.max(delta,noise[label-1]+noise[ngblb-1]);
                }
            }
        } else
	    if (outputs.equals("weights")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                float wgty = Numerics.bounded((inputImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta)
	                        *Numerics.bounded((interior[label-1]-inputImage[xyz])/(interior[label-1]-interior[ngblb-1]), delta, 1.0f-delta);
                float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelset[xyz]/dist0)), delta, 1.0f-delta);
                    
                memsImage[xyz] = wgtx*wgty;
            }
        } else
        /* no offset for simplicity
	    if (outputs.equals("offset")) {	    
            for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
                int ngblb = neighbor[xyz];
                
                if (levelset[xyz]<1.0f) {
                    // find the neighbor in the list
                    int loc = -1;
                    for (int n=0;n<nngb;n++) {
                         if (ngblb==ngblist[n][label-1]) {
                             loc=n;
                             n=nngb;
                         }
                    }
                    if (loc>-1)
                        memsImage[xyz] = offset[loc][label-1];
                }
            }
        } else
        */
	    if (outputs.equals("slope")) {	    
	        for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
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
                    if (loc>-1)
                        memsImage[xyz] = slope[label-1][loc];
                }
            }
        } else
	    if (outputs.equals("posterior")) {	    
	        for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
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
                    if (loc>-1) {
                        // probability of existence: CNR>0.5, N_boundary>scaling
                        double pcnr = 1.0-FastMath.exp(-0.5f*Numerics.square(4.0f*(interior[label-1]-interior[ngblb-1])/Numerics.max(delta,noise[label-1]+noise[ngblb-1])));
                        double pbnd = 1.0-FastMath.exp(-0.5f*Numerics.square(bdcount[label-1][loc]/scaling));
                        memsImage[xyz] = (float)FastMath.sqrt(pcnr*pbnd);
                    }
                }
            }
        } else
	    if (outputs.equals("sharpness")) {	    
	        for (int xyz=0;xyz<nxyz;xyz++) if (parcelImage[xyz]>0 && neighbor[xyz]>0) {
                int label = parcelImage[xyz];
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
                    if (loc>-1) {
                        // probability of existence: CNR>0.5, N_boundary>scaling
                        double pcnr = 1.0-FastMath.exp(-0.5f*Numerics.square(4.0f*(interior[label-1]-interior[ngblb-1])/Numerics.max(delta,noise[label-1]+noise[ngblb-1])));
                        double pbnd = 1.0-FastMath.exp(-0.5f*Numerics.square(bdcount[label-1][loc]/scaling));
                        if (pcnr*pbnd>0.25) {
                            memsImage[xyz] = Numerics.abs(slope[label-1][loc]);
                        }
                    }
                }
            }
        }
	    return;
	}

}
