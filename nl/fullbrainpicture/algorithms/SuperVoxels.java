package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;


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
	    float[][] centroid = new float[3][nsxyz];
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        centroid[X][xyzs] = xs*scaling + 0.5f*scaling;
	        centroid[Y][xyzs] = ys*scaling + 0.5f*scaling;
	        centroid[Z][xyzs] = zs*scaling + 0.5f*scaling;
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
	    // start a voxel heap at each center
		BinaryHeap4D heap = new BinaryHeap4D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MINTREE);
		boolean[] processed = new boolean[nx*ny*nz];
		for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        int x0 = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y0 = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z0 = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz0 = x0+nx*y0+nx*ny*z0;
	        if (mask[xyz0]) {
	            // set as starting point
	            parcelImage[xyz0] = xyzs+1;
	            rescaledImage[xyzs] = inputImage[xyz0];
	            count[xyzs] = 1;
	            processed[xyz0] = true;
	            
	            // add neighbors to the tree
	            for (byte k = 0; k<6; k++) {
	                switch (k) {
	                    case 0: x0++;
	                    case 1: x0--;
	                    case 2: y0++;
	                    case 3: y0--;
	                    case 4: z0++;
	                    case 5: z0--;
                    }
                    // exclude zero as mask
                    if (mask[x0+nx*y0+nx*ny*z0]) {
                        
                        // distance function
                        float dist = (x0-centroid[X][xyzs])*(x0-centroid[X][xyzs])
                                    +(y0-centroid[Y][xyzs])*(y0-centroid[Y][xyzs])
                                    +(z0-centroid[Z][xyzs])*(z0-centroid[Z][xyzs]);
                                    
                        float contrast = (inputImage[x0+nx*y0+nx*ny*z0]-rescaledImage[xyzs])
                                        *(inputImage[x0+nx*y0+nx*ny*z0]-rescaledImage[xyzs]);
                                        
                        heap.addValue(noise*dist+contrast, x0,y0,z0, xyzs+1);
                    }
                    switch (k) {
	                    case 0: x0--;
	                    case 1: x0++;
	                    case 2: y0--;
	                    case 3: y0++;
	                    case 4: z0--;
	                    case 5: z0++;
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
            for (byte k = 0; k<6; k++) {
                switch (k) {
                    case 0: if (x<nx-1) x++;
                    case 1: if (x>0) x--;
                    case 2: if (y<ny-1) y++;
                    case 3: if (y>0) y--;
                    case 4: if (z<nz-1) z++;
                    case 5: if (z>0) z--;
                }
                // exclude zero as mask
                if (mask[x+nx*y+nx*ny*z] && !processed[x+nx*y+nx*ny*z]) {
                    
                    // distance function
                    float dist = (x-centroid[X][xyzs])*(x-centroid[X][xyzs])
                                +(y-centroid[Y][xyzs])*(y-centroid[Y][xyzs])
                                +(z-centroid[Z][xyzs])*(z-centroid[Z][xyzs]);
                                
                    float contrast = (inputImage[x+nx*y+nx*ny*z]-rescaledImage[xyzs])
                                    *(inputImage[x+nx*y+nx*ny*z]-rescaledImage[xyzs]);
                                    
                    heap.addValue(noise*dist+contrast, x,y,z, xyzs+1);
                }
                switch (k) {
                    case 0: if (x>0) x--;
                    case 1: if (x<nx-1) x++;
                    case 2: if (y>0) y--;
                    case 3: if (y<ny-1) y++;
                    case 4: if (z>0) z--;
                    case 5: if (z<nz-1) z++;
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
		}else if (outputs.equals("sharpness")) {
		    growBoundaries();
		    fitBoundarySigmoid();
		    memsImage = new float[nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) {
		        if (parcelImage[xyz]>0) {
		            memsImage[xyz] = levelset[xyz];
		        }
		    }
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
                    if (cmask[xyzn] && parcelImage[xyzn]!=parcelImage[xyz]) {
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
	    float[] interior = new float[nsxyz];
	    float[] incount = new float[nsxyz];
	    
	    // boundaries
	    for (int xyz=0;xyz<nx;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        
	        interior[label-1] += levelset[xyz]*inputImage[xyz];
	        incount[label-1] += levelset[xyz];
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        if (incount[xyzs]>0) interior[xyzs] /= incount[xyzs];
	    }
	    
	    float[][] slope = new float[3][nsxyz];
	    float[][] offset = new float[3][nsxyz];
	    float[][] bdcount = new float[3][nsxyz];

	    // offset
	    for (int xyz=0;xyz<nx;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	    
	        float weight = Numerics.bounded((parcelImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                      *Numerics.bounded((interior[label-1]-parcelImage[xyz])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f);
	        if (ngblb==label+1) {
	            offset[X][label-1] += levelset[xyz]*weight;
	            bdcount[X][label-1] += weight;                                          
	        } else if (ngblb==label+nsx) {
	            offset[Y][label-1] += levelset[xyz]*weight;
	            bdcount[Y][label-1] += weight;                                          
	        } else if (ngblb==label+nsx*nsy) {
	            offset[Z][label-1] += levelset[xyz]*weight;
	            bdcount[Z][label-1] += weight;                                          
	        } else if (ngblb==label-1) {
	            offset[X][ngblb-1] -= levelset[xyz]*weight;
	            bdcount[X][ngblb-1] += weight;   
	        } else if (ngblb==label-nsx) {
	            offset[Y][ngblb-1] -= levelset[xyz]*weight;
	            bdcount[Y][ngblb-1] += weight;                                          
	        } else if (ngblb==label-nsx*nsy) {
	            offset[Z][ngblb-1] -= levelset[xyz]*weight;
	            bdcount[Z][ngblb-1] += weight;                                          
	        }
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<3;c++) {
	        if (bdcount[c][xyzs]>0) offset[c][xyzs] /= bdcount[c][xyzs];
	    }
	    
	    // slope
	    for (int xyz=0;xyz<nx;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	        
	        float weight = Numerics.bounded((parcelImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                      *Numerics.bounded((interior[label-1]-parcelImage[xyz])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f);
	        if (ngblb==label+1) {
	            slope[X][label-1] += Numerics.bounded((parcelImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[X][xyz])*weight;
	        } else if (ngblb==label+nsx) {
	            slope[Y][label-1] += Numerics.bounded((parcelImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[Y][xyz])*weight;
	        } else if (ngblb==label+nsx*nsy) {
	            slope[Z][label-1] += Numerics.bounded((parcelImage[xyz]-interior[ngblb-1])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[Z][xyz])*weight;
	        } else if (ngblb==label-1) {
	            slope[X][ngblb-1] += Numerics.bounded((interior[label-1]-parcelImage[xyz])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[X][xyz])*weight;
	        } else if (ngblb==label-nsx) {
	            slope[Y][ngblb-1] += Numerics.bounded((interior[label-1]-parcelImage[xyz])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[Y][xyz])*weight;
	        } else if (ngblb==label-nsx*nsy) {
	            slope[Z][ngblb-1] += Numerics.bounded((interior[label-1]-parcelImage[xyz])/(interior[label-1]-interior[ngblb-1]), 0.0f, 1.0f)
	                                /(levelset[xyz]-offset[Z][xyz])*weight;
	        }
	    }
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) for (int c=0;c<3;c++) {
	        if (bdcount[c][xyzs]>0) slope[c][xyzs] /= bdcount[c][xyzs];
	    }
	        
	    // outputs?
	    memsImage = new float[nxyz];
		    
	    for (int xyz=0;xyz<nx;xyz++) if (parcelImage[xyz]>0) {
	        int label = parcelImage[xyz];
	        int ngblb = neighbor[xyz];
	        
	        if (levelset[xyz]<1.0f) {
	            memsImage[xyz] = Numerics.max(slope[X][label-1],slope[Y][label-1],slope[Z][label-1],
	                                          slope[X][ngblb-1],slope[Y][ngblb-1],slope[Z][ngblb-1]);
	        }
	    }
	    return;
	}

}
