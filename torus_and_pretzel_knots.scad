//Authors: Sean O'Neil and Laura Taalman
//Code to generate models for torus and pretzel knots
//Made for UnKnot III

////////////////////////////////////////////////////////////////////

// Global parameters 

$fn = 24;		// use $fn=12 for draft, $fn=24 for final
step = 0.005;	// use 0.005 for draft, 0.001 for final

modelscale = 10; //Scale of the model
width = 3; //Width of the tubes

function shape() = circle(width); //Can change to other shapes if desired, though the corners have spheres

////////////////////////////////////////////////////////////////////

//Torus knot (link) variables. Note that if p and q are not coprime the resulting model will have more than one component.

p = 2;
q = 5;

//Uncomment for (p, q)-torus knot
renderTorus(p, q);

////////////////////////////////////////////////////////////////////

//Pretzel knot (and maybe later general tangle) variables

gapsize = 10; //Space between the twists
padding = 5; //Space between the ends of the twists and the connection to the other twists
twistsize = 4; //Radius of the twist columns
twistlength = 6; //Length of a half-twist

//(Also includes pretzel links)
//These take a lot longer to render than the pretzel links..
//I recommend rotating the model so that the twists are vertical before printing
//First parameter is a vector for number of twists, second parameter is a vector of the signs of each twist region

//Uncomment for flat pretzel knot
//renderPretzelFlat([3, 3, 3], [1, -1, 1]);

//Uncomment for flat pretzel knot where all twist regions are the same height (extra parameter is height of all the twists)
//renderPretzelStretchedFlat([2, 3, 7], [-1, 1, 1], 20);

//Uncomment for symmetric pretzel knot
//renderPretzelCompact([3, 3, 3], [1, 1, 1]);

//Uncomment for symmetric pretzel knot where all twist regions are the same height (extra parameter is height of all the twists)
//renderPretzelStretchedCompact([2, 3, 7], [-1, 1, 1], 20);


////////////////////////////////////////////////////////////////////

//Torus knot functions

//Torus knot parametric functions

function Tx(theta) = ((cos(q*theta) + 2)*cos(p*theta));
function Ty(theta) = ((cos(q*theta) + 2)*sin(p*theta));
function Tz(theta) = -sin(q*theta);

//Computes the gcd of m and n, recursively

function gcd(m, n) = (abs(m) >= abs(n)) ? ( (abs(m) % abs(n) == 0) ? abs(n) : gcd(abs(n), abs(m) % abs(n)) ) : ( (abs(n) % abs(m) == 0) ? abs(m) : gcd(abs(m), abs(n) % abs(m)) );

//Torus knot rendering function

module renderTorus(p, q) {
    gcd = gcd(p, q); //Note: gcd(p, q) is the number of components of the link.
    for (c = [0 : gcd-1]) {
        path = [for (t=[0:step:1]) modelscale*rotate(Tx(t*360), Ty(t*360), Tz(t*360), 360*c/q)];
        path_transforms = construct_transform_path(path, true);
        sweep(shape(), path_transforms, true);
    }
}

//Rotates about the z axis by phi, simple enough

function rotate(x, y, z, phi) = [sqrt(x*x + y*y)*cos(atan2(y, x) + phi), sqrt(x*x + y*y)*sin(atan2(y, x) + phi), z];

////////////////////////////////////////////////////////////////////

//Pretzel knot functions

//Renders a flat pretzel knot

module renderPretzelFlat(twists, signs) {
    for (i = [0 : 1 : len(twists)-1]) {
        renderTwist(-gapsize*(len(twists)-1)/2 + gapsize*i, 0, 0, 0, twists[i], sign(signs[i]));
    }
    for (i = [0 : 1 : len(twists)-2]) {
        renderPadding(-gapsize*(len(twists)-1)/2 + gapsize*i + twistsize, twists[i]/2*twistlength, 0, -gapsize*(len(twists)-1)/2 + gapsize*(i+1) - twistsize, twists[i+1]/2*twistlength, 0, 0, 0, 1, padding);
        renderPadding(-gapsize*(len(twists)-1)/2 + gapsize*i + twistsize, -twists[i]/2*twistlength, 0, -gapsize*(len(twists)-1)/2 + gapsize*(i+1) - twistsize, -twists[i+1]/2*twistlength, 0, 0, 0, 0, padding);
    }
    padding = twistlength/2*(1 + max(twists) - max(twists[0],twists[len(twists)-1]));
    renderPadding(-gapsize*(len(twists)-1)/2 - twistsize, twists[0]/2*twistlength, 0, -gapsize*(len(twists)-1)/2 + gapsize*(len(twists)-1) + twistsize, twists[len(twists)-1]/2*twistlength, 0, 0, 1, 1, padding);
    renderPadding(-gapsize*(len(twists)-1)/2 - twistsize, -twists[0]/2*twistlength, 0, -gapsize*(len(twists)-1)/2 + gapsize*(len(twists)-1) + twistsize, -twists[len(twists)-1]/2*twistlength, 0, 0, 1, 0, padding);
}

//Renders a flat pretzel knot such that the twists are all the same height

module renderPretzelStretchedFlat(twists, signs, twistheight) {
    for (i = [0 : 1 : len(twists)-1]) {
        renderTwistStretched(-gapsize*(len(twists)-1)/2 + gapsize*i, 0, 0, 0, twists[i], sign(signs[i]), twistheight);
    }
    for (i = [0 : 1 : len(twists)-2]) {
        renderPadding(-gapsize*(len(twists)-1)/2 + gapsize*i + twistsize, twistheight/2, 0, -gapsize*(len(twists)-1)/2 + gapsize*(i+1) - twistsize, twistheight/2, 0, 0, 0, 1, padding);
        renderPadding(-gapsize*(len(twists)-1)/2 + gapsize*i + twistsize, -twistheight/2, 0, -gapsize*(len(twists)-1)/2 + gapsize*(i+1) - twistsize, -twistheight/2, 0, 0, 0, 0, padding);
    }
    renderPadding(-gapsize*(len(twists)-1)/2 - twistsize, twistheight/2, 0, -gapsize*(len(twists)-1)/2 + gapsize*(len(twists)-1) + twistsize, twistheight/2, 0, 0, 1, 1, padding);
    renderPadding(-gapsize*(len(twists)-1)/2 - twistsize, -twistheight/2, 0, -gapsize*(len(twists)-1)/2 + gapsize*(len(twists)-1) + twistsize, -twistheight/2, 0, 0, 1, 0, padding);
}

//Renders a more symmetric conformation of a pretzel knot

module renderPretzelCompact(twists, signs) {
    //echo("twists");
    for (i = [0 : 1 : len(twists)-1]) {
        phi = 360*i/len(twists);
        renderTwistCompact(gapsize*cos(phi), 0, gapsize*sin(phi), 0, twists[i], sign(signs[i]), phi);
    }
    for (i = [0 : 1 : len(twists)-2]) {
        phi0 = 360*i/len(twists);
        phi1 = 360*(i+1)/len(twists);
        renderPadding( (gapsize + twistsize)*cos(phi0), twists[i]/2*twistlength, (gapsize + twistsize)*sin(phi0), (gapsize - twistsize)*cos(phi1), twists[i+1]/2*twistlength, (gapsize - twistsize)*sin(phi1), 0, 1, 1, padding);
        renderPadding((gapsize + twistsize)*cos(phi0), -twists[i]/2*twistlength, (gapsize + twistsize)*sin(phi0), (gapsize - twistsize)*cos(phi1), -twists[i+1]/2*twistlength, (gapsize - twistsize)*sin(phi1), 0, 1, 0, padding);
    }
    phi = 360*(len(twists)-1)/len(twists);
    padding = twistlength/2*(1 + max(twists) - max(twists[0],twists[len(twists)-1]));
    renderPadding(gapsize - twistsize, twists[0]/2*twistlength, 0, (gapsize + twistsize)*cos(phi), twists[len(twists)-1]/2*twistlength, (gapsize + twistsize)*sin(phi), 0, 1, 1, padding);
    renderPadding(gapsize - twistsize, -twists[0]/2*twistlength, 0, (gapsize + twistsize)*cos(phi), -twists[len(twists)-1]/2*twistlength, (gapsize + twistsize)*sin(phi), 0, 1, 0, padding);
}

//Renders a more symmetric conformation of a pretzel knot such that the twists are all the same height

module renderPretzelStretchedCompact(twists, signs, twistheight) {
    for (i = [0 : 1 : len(twists)-1]) {
        phi = 360*i/len(twists);
        renderTwistStretchedCompact(gapsize*cos(phi), 0, gapsize*sin(phi), 0, twists[i], sign(signs[i]), phi, twistheight);
    }
    for (i = [0 : 1 : len(twists)-2]) {
        phi0 = 360*i/len(twists);
        phi1 = 360*(i+1)/len(twists);
        renderPadding( (gapsize + twistsize)*cos(phi0), twistheight/2, (gapsize + twistsize)*sin(phi0), (gapsize - twistsize)*cos(phi1), twistheight/2, (gapsize - twistsize)*sin(phi1), 0, 1, 1, padding);
        renderPadding((gapsize + twistsize)*cos(phi0), -twistheight/2, (gapsize + twistsize)*sin(phi0), (gapsize - twistsize)*cos(phi1), -twistheight/2, (gapsize - twistsize)*sin(phi1), 0, 1, 0, padding);
    }
    phi = 360*(len(twists)-1)/len(twists);
    renderPadding(gapsize - twistsize, twistheight/2, 0, (gapsize + twistsize)*cos(phi), twistheight/2, (gapsize + twistsize)*sin(phi), 0, 1, 1, padding);
    renderPadding(gapsize - twistsize, -twistheight/2, 0, (gapsize + twistsize)*cos(phi), -twistheight/2, (gapsize + twistsize)*sin(phi), 0, 1, 0, padding);
}

////////////////////////////////////////////////////////////////////

//Twist functions for pretzel knots and tangle knots (incomplete, only has one orientation for tangles at the moment)

//Renders a standard twist region

module renderTwist(x, y, z, orientation, count, sgn) {
    if (orientation == 0) {
        //+
        x0 = x;
        y0 = y - count/2*twistlength;
        y1 = y + count/2*twistlength;
        z0 = z;
        delta = step; //I don't know why, but this is necessary or the model will be disconnected.
        path = [for (t=[0:step:1+delta]) modelscale*[x0 + twistsize*cos(180*t*count), y0 + (y1-y0)*t, z0 - sgn*twistsize*sin(180*t*count)]];
        path_transforms = construct_transform_path(path, false);
        sweep(shape(), path_transforms, false);
        path2 = [for (t=[0:step:1+delta]) modelscale*[x0 - twistsize*cos(180*t*count), y0 + (y1-y0)*t, z0 + sgn*twistsize*sin(180*t*count)]];
        path_transforms2 = construct_transform_path(path2, false);
        sweep(shape(), path_transforms2, false);
    }
    else if (orientation == 1) {
        //x
    }
}

//Renders a twist region with a given height

module renderTwistStretched(x, y, z, orientation, count, sgn, twistheight) {
    if (orientation == 0) {
        //+
        x0 = x;
        y0 = y - twistheight/2;
        y1 = y + twistheight/2;
        z0 = z;
        delta = step; //I don't know why, but this is necessary or the model will be disconnected.
        path = [for (t=[0:step:1+delta]) modelscale*[x0 + twistsize*cos(180*t*count), y0 + (y1-y0)*t, z0 - sgn*twistsize*sin(180*t*count)]];
        path_transforms = construct_transform_path(path, false);
        sweep(shape(), path_transforms, false);
        path2 = [for (t=[0:step:1+delta]) modelscale*[x0 - twistsize*cos(180*t*count), y0 + (y1-y0)*t, z0 + sgn*twistsize*sin(180*t*count)]];
        path_transforms2 = construct_transform_path(path2, false);
        sweep(shape(), path_transforms2, false);
    }
    else if (orientation == 1) {
        //x
    }
}

//Renders a twist region rotated by a given angle

module renderTwistCompact(x, y, z, orientation, count, sgn, angle) {
    if (orientation == 0) {
        //+
        x0 = x;
        y0 = y - count/2*twistlength;
        y1 = y + count/2*twistlength;
        z0 = z;
        delta = step; //I don't know why, but this is necessary or the model will be disconnected.
        path = [for (t=[0:step:1+delta]) modelscale*[x0 + twistsize*cos(-angle + 180*t*count), y0 + (y1-y0)*t, z0 - sgn*twistsize*sin(-angle + 180*t*count)]];
        path_transforms = construct_transform_path(path, false);
        sweep(shape(), path_transforms, false);
        path2 = [for (t=[0:step:1+delta]) modelscale*[x0 - twistsize*cos(-angle + 180*t*count), y0 + (y1-y0)*t, z0 + sgn*twistsize*sin(-angle + 180*t*count)]];
        path_transforms2 = construct_transform_path(path2, false);
        sweep(shape(), path_transforms2, false);
    }
    else if (orientation == 1) {
        //x
    }
}

//Renders a twist region with a given height rotated by a given angle

module renderTwistStretchedCompact(x, y, z, orientation, count, sgn, angle, twistheight) {
    if (orientation == 0) {
        //+
        x0 = x;
        y0 = y - twistheight/2;
        y1 = y + twistheight/2;
        z0 = z;
        delta = step; //I don't know why, but this is necessary or the model will be disconnected.
        path = [for (t=[0:step:1+delta]) modelscale*[x0 + twistsize*cos(-angle + 180*t*count), y0 + (y1-y0)*t, z0 - sgn*twistsize*sin(-angle + 180*t*count)]];
        path_transforms = construct_transform_path(path, false);
        sweep(shape(), path_transforms, false);
        path2 = [for (t=[0:step:1+delta]) modelscale*[x0 - twistsize*cos(-angle + 180*t*count), y0 + (y1-y0)*t, z0 + sgn*twistsize*sin(-angle + 180*t*count)]];
        path_transforms2 = construct_transform_path(path2, false);
        sweep(shape(), path_transforms2, false);
    }
    else if (orientation == 1) {
        //x
    }
}

////////////////////////////////////////////////////////////////////

//Padding functions for pretzel knots and tangle knots (incomplete, only works for one tangle orientation at the moment)

//Renders the standard connection between two twists with some padding

module renderPadding(x0, y0, z0, x1, y1, z1, orientation, nestlevel, side, pad) {
    //Go out to padding length * nestlevel
    if (orientation == 0) {
        //+
        if (side == 1) {
            segment(x0, y0, z0, x0, max(y0, y1)+nestlevel*pad, z0);
            segment(x1, y1, z1, x1, max(y0, y1)+nestlevel*pad, z1);
            segment2(x0, max(y0, y1)+nestlevel*pad, z0, x1, max(y0, y1)+nestlevel*pad, z1);
        }
        else if (side == 0) {
            segment(x0, y0, z0, x0, min(y0, y1)-nestlevel*pad, z0);
            segment(x1, y1, z1, x1, min(y0, y1)-nestlevel*pad, z1);
            segment2(x0, min(y0, y1)-nestlevel*pad, z0, x1, min(y0, y1)-nestlevel*pad, z1);
        }
    }
    else if (orientation == 1) {
        //x
    }
}

////////////////////////////////////////////////////////////////////

//Segment functions for constructing the padding connections

//Creates a cylinder pointing from (x0, y0, z0) to (x1, y1, z1), with a sphere at the beginning.

module segment(x0, y0, z0, x1, y1, z1){
    delta = step; //I don't know why, but this is necessary or the model will be disconnected.
    path = [for (t=[0:step:1+delta]) modelscale*[x0 + (x1-x0)*t, y0 + (y1-y0)*t, z0 + (z1-z0)*t]];
	path_transforms = construct_transform_path(path, false);
	sweep(shape(), path_transforms, false);
    translate(modelscale*[x0, y0, z0])
        sphere(width, $fn);
}


//Same as previous segment function, but adds another sphere onto the end, aesthetic or necessary depending on how the model is actually made.

module segment2(x0, y0, z0, x1, y1, z1){
    delta = step; //I don't know why, but this is necessary or the model will be disconnected.
    path = [for (t=[0:step:1+delta]) modelscale*[x0 + (x1-x0)*t, y0 + (y1-y0)*t, z0 + (z1-z0)*t]];
	path_transforms = construct_transform_path(path, false);
	sweep(shape(), path_transforms, false);
    translate(modelscale*[x0, y0, z0])
        sphere(width, $fn);
    translate(modelscale*[x1, y1, z1])
        sphere(width, $fn);
}

////////////////////////////////////////////////////////////////////
//Don't edit anything below here.
//Thanks to Laura Taalman for figuring out the sweeper stuff.

// mathgrrl sweeper curves

// ugh lots of included stuff from scad-utils 
// also of course the sweeper code

/*
////////////////////////////////////////////////////////////////////
// function
 
function f(t) =  
 [ cos(5*t)*cos(4*t), 	// CHANGE THE FIVE NUMBERS
   cos(5*t)*sin(4*t), 	// AND SEE WHAT HAPPENS WHEN YOU PRESS F5
   sin(5*t)
 ]; 

sweeper();
*/
////////////////////////////////////////////////////////////////////
// module for sweeping out the curve

/*
module sweeper(){
	path = [for (t=[0:step:1]) ball_radius*f(t*360)];
	path_transforms = construct_transform_path(path,true);
	sweep(shape(), path_transforms,true);
}
*/

// ball: 777 44
// interesting: 332 22
// interesting: 331 44
// knotted(!): 335 44

/* default
function f(t) =  
 [ cos(7*t)*cos(4*t), 	// CHANGE THE FIVE NUMBERS
   cos(7*t)*sin(4*t), 	// AND SEE WHAT HAPPENS WHEN YOU PRESS F5
   sin(7*t)
 ]; 
*/

/* pretty
function f(t) =  
 [ (1+cos(4*t))*cos(3*t), 	// CHANGE THE FIVE NUMBERS
   cos(4*t)*sin(3*t),   	// AND SEE WHAT HAPPENS WHEN YOU PRESS F5
   sin(4*t)
 ]; 
*/

/* wicked
function f(t) =  
 [ (1.5+cos(4*t))*cos(19*t), 	// CHANGE THE FIVE NUMBERS
   cos(2*t)*sin(27*t), 	// AND SEE WHAT HAPPENS WHEN YOU PRESS F5
   2*cos(4*t)
 ]; 
*/

/* TMAT printed
function f(t) =  
 [ (1+cos(3*t))*cos(2*t), 
   (1+cos(3*t))*sin(2*t), 
   sin(3*t)
 ]; 
*/

/* TMAT printed - tall torus spring - wire 4 ball 30
function f(t) =  
 [ (2+cos(4*t))*cos(17*t), 
   (2+cos(4*t))*sin(17*t), 
   5*sin(4*t)
 ]; 
 */
 
 /*
 Another tall torus knot
 function f(t) =  
 [ (1.5+cos(3*t))*cos(11*t), 
   (1.5+cos(3*t))*sin(11*t), 
   4*sin(3*t) 
 ]; 
 */

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// SHAPES

function square(size) = [[-size,-size], [-size,size], [size,size], [size,-size]] / 2;

function circle(r) = [for (i=[0:$fn-1]) let (a=i*360/$fn) r * [cos(a), sin(a)]];

function regular(r, n) = circle(r, $fn=n);

function rectangle_profile(size=[1,1]) = [	
	// The first point is the anchor point, put it on the point corresponding to [cos(0),sin(0)]
	[ size[0]/2,  0], 
	[ size[0]/2,  size[1]/2],
	[-size[0]/2,  size[1]/2],
	[-size[0]/2, -size[1]/2],
	[ size[0]/2, -size[1]/2],
];

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// SWEEPER 
 
//use <scad-utils/linalg.scad>
//use <scad-utils/transformations.scad>
//use <scad-utils/lists.scad>

function rotation_from_axis(x,y,z) = [[x[0],y[0],z[0]],[x[1],y[1],z[1]],[x[2],y[2],z[2]]]; 

function rotate_from_to(a,b,_axis=[]) = 
        len(_axis) == 0 
        ? rotate_from_to(a,b,unit(cross(a,b))) 
        : _axis*_axis >= 0.99 ? rotation_from_axis(unit(b),_axis,cross(_axis,unit(b))) * 
    transpose_3(rotation_from_axis(unit(a),_axis,cross(_axis,unit(a)))) : identity3(); 

function make_orthogonal(u,v) = unit(u - unit(v) * (unit(v) * u)); 

// Prevent creeping nonorthogonality 
function coerce(m) = [unit(m[0]), make_orthogonal(m[1],m[0]), make_orthogonal(make_orthogonal(m[2],m[0]),m[1])]; 

function tangent_path(path, i) =
i == 0 ?
  unit(path[1] - path[0]) :
  (i == len(path)-1 ?
      unit(path[i] - path[i-1]) :
    unit(path[i+1]-path[i-1]));

function construct_rt(r,t) = [concat(r[0],t[0]),concat(r[1],t[1]),concat(r[2],t[2]),[0,0,0,1]]; 

function construct_torsion_minimizing_rotations(tangents) = [ 
        for (i = [0:len(tangents)-2]) 
                rotate_from_to(tangents[i],tangents[i+1]) 
]; 

function accumulate_rotations(rotations,acc_=[]) = let(i = len(acc_)) 
        i ==  len(rotations) ? acc_ : 
        accumulate_rotations(rotations, 
                i == 0 ? [rotations[0]] : concat(acc_, [ rotations[i] * acc_[i-1] ]) 
        ); 

// Calculates the relative torsion along the Z axis for two transformations 
function calculate_twist(A,B) = let( 
        D = transpose_3(B) * A 
) atan2(D[1][0], D[0][0]); 
        
function construct_transform_path(path, closed=false) = let( 
        l = len(path), 
        tangents = [ for (i=[0:l-1]) tangent_path(path, i)], 
        local_rotations = construct_torsion_minimizing_rotations(concat([[0,0,1]],tangents)), 
        rotations = accumulate_rotations(local_rotations), 
        twist = closed ? calculate_twist(rotations[0], rotations[l-1]) : 0 
)  [ for (i = [0:l-1]) construct_rt(rotations[i], path[i]) * rotation([0,0,twist*i/(l-1)])]; 

module sweep(shape, path_transforms, closed=false) {

    pathlen = len(path_transforms);
    segments = pathlen + (closed ? 0 : -1);
    shape3d = to_3d(shape);

    function sweep_points() =
      flatten([for (i=[0:pathlen]) transform(path_transforms[i], shape3d)]);

    function loop_faces() = [let (facets=len(shape3d))
        for(s=[0:segments-1], i=[0:facets-1])
          [(s%pathlen) * facets + i, 
           (s%pathlen) * facets + (i + 1) % facets, 
           ((s + 1) % pathlen) * facets + (i + 1) % facets, 
           ((s + 1) % pathlen) * facets + i]];

    bottom_cap = closed ? [] : [[for (i=[len(shape3d)-1:-1:0]) i]];
    top_cap = closed ? [] : [[for (i=[0:len(shape3d)-1]) i+len(shape3d)*(pathlen-1)]];
    polyhedron(points = sweep_points(), faces = concat(loop_faces(), bottom_cap, top_cap), convexity=5);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// LINALG 

// very minimal set of linalg functions needed by so3, se3 etc.

// cross and norm are builtins
//function cross(x,y) = [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]];
//function norm(v) = sqrt(v*v);

function vec3(p) = len(p) < 3 ? concat(p,0) : p;
function vec4(p) = let (v3=vec3(p)) len(v3) < 4 ? concat(v3,1) : v3;
function unit(v) = v/norm(v);

function identity3()=[[1,0,0],[0,1,0],[0,0,1]]; 
function identity4()=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];


function take3(v) = [v[0],v[1],v[2]];
function tail3(v) = [v[3],v[4],v[5]];
function rotation_part(m) = [take3(m[0]),take3(m[1]),take3(m[2])];
function rot_trace(m) = m[0][0] + m[1][1] + m[2][2];
function rot_cos_angle(m) = (rot_trace(m)-1)/2;

function rotation_part(m) = [take3(m[0]),take3(m[1]),take3(m[2])];
function translation_part(m) = [m[0][3],m[1][3],m[2][3]];
function transpose_3(m) = [[m[0][0],m[1][0],m[2][0]],[m[0][1],m[1][1],m[2][1]],[m[0][2],m[1][2],m[2][2]]];
function transpose_4(m) = [[m[0][0],m[1][0],m[2][0],m[3][0]],
                           [m[0][1],m[1][1],m[2][1],m[3][1]],
                           [m[0][2],m[1][2],m[2][2],m[3][2]],
                           [m[0][3],m[1][3],m[2][3],m[3][3]]]; 
function invert_rt(m) = construct_Rt(transpose_3(rotation_part(m)), -(transpose_3(rotation_part(m)) * translation_part(m)));
function construct_Rt(R,t) = [concat(R[0],t[0]),concat(R[1],t[1]),concat(R[2],t[2]),[0,0,0,1]];

// Hadamard product of n-dimensional arrays
function hadamard(a,b) = !(len(a)>0) ? a*b : [ for(i = [0:len(a)-1]) hadamard(a[i],b[i]) ];

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// TRANSFORMATIONS

/*!
  Creates a rotation matrix

  xyz = euler angles = rz * ry * rx
  axis = rotation_axis * rotation_angle
*/
function rotation(xyz=undef, axis=undef) = 
	xyz != undef && axis != undef ? undef :
	xyz == undef  ? se3_exp([0,0,0,axis[0],axis[1],axis[2]]) :
	len(xyz) == undef ? rotation(axis=[0,0,xyz]) :
	(len(xyz) >= 3 ? rotation(axis=[0,0,xyz[2]]) : identity4()) *
	(len(xyz) >= 2 ? rotation(axis=[0,xyz[1],0]) : identity4()) *
	(len(xyz) >= 1 ? rotation(axis=[xyz[0],0,0]) : identity4());

/*!
  Creates a scaling matrix
*/
function scaling(v) = [
	[v[0],0,0,0],
	[0,v[1],0,0],
	[0,0,v[2],0],
	[0,0,0,1],
];

/*!
  Creates a translation matrix
*/
function translation(v) = [
	[1,0,0,v[0]],
	[0,1,0,v[1]],
	[0,0,1,v[2]],
	[0,0,0,1],
];

// Convert between cartesian and homogenous coordinates
function project(x) = subarray(x,end=len(x)-1) / x[len(x)-1];

function transform(m, list) = [for (p=list) project(m * vec4(p))];
function to_3d(list) = [ for(v = list) vec3(v) ];

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// LISTS

// List helpers

/*!
  Flattens a list one level:

  flatten([[0,1],[2,3]]) => [0,1,2,3]
*/
function flatten(list) = [ for (i = list, v = i) v ];


/*!
  Creates a list from a range:

  range([0:2:6]) => [0,2,4,6]
*/
function range(r) = [ for(x=r) x ];

/*!
  Reverses a list:

  reverse([1,2,3]) => [3,2,1]
*/
function reverse(list) = [for (i = [len(list)-1:-1:0]) list[i]];

/*!
  Extracts a subarray from index begin (inclusive) to end (exclusive)
  FIXME: Change name to use list instead of array?

  subarray([1,2,3,4], 1, 2) => [2,3]
*/
function subarray(list,begin=0,end=-1) = [
    let(end = end < 0 ? len(list) : end)
      for (i = [begin : 1 : end-1])
        list[i]
];

/*!
  Returns a copy of a list with the element at index i set to x

  set([1,2,3,4], 2, 5) => [1,2,5,4]
*/
function set(list, i, x) = [for (i_=[0:len(list)-1]) i == i_ ? x : list[i_]];
    
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// SE3

function combine_se3_exp(w, ABt) = construct_Rt(rodrigues_so3_exp(w, ABt[0], ABt[1]), ABt[2]);

// [A,B,t]
function se3_exp_1(t,w) = concat(
	so3_exp_1(w*w),
	[t + 0.5 * cross(w,t)]
);

function se3_exp_2(t,w) = se3_exp_2_0(t,w,w*w);
function se3_exp_2_0(t,w,theta_sq) = 
se3_exp_23(
	so3_exp_2(theta_sq), 
	C = (1.0 - theta_sq/20) / 6,
	t=t,w=w);

function se3_exp_3(t,w) = se3_exp_3_0(t,w,sqrt(w*w)*180/PI,1/sqrt(w*w));

function se3_exp_3_0(t,w,theta_deg,inv_theta) = 
se3_exp_23(
	so3_exp_3_0(theta_deg = theta_deg, inv_theta = inv_theta),
	C = (1 - sin(theta_deg) * inv_theta) * (inv_theta * inv_theta),
	t=t,w=w);

function se3_exp_23(AB,C,t,w) = 
[AB[0], AB[1], t + AB[1] * cross(w,t) + C * cross(w,cross(w,t)) ];

function se3_exp(mu) = se3_exp_0(t=take3(mu),w=tail3(mu)/180*PI);

function se3_exp_0(t,w) =
combine_se3_exp(w,
// Evaluate by Taylor expansion when near 0
	w*w < 1e-8 
	? se3_exp_1(t,w)
	: w*w < 1e-6
	  ? se3_exp_2(t,w)
	  : se3_exp_3(t,w)
);

function se3_ln(m) = se3_ln_to_deg(se3_ln_rad(m));
function se3_ln_to_deg(v) = concat(take3(v),tail3(v)*180/PI);

function se3_ln_rad(m) = se3_ln_0(m, 
	rot = so3_ln_rad(rotation_part(m)));
function se3_ln_0(m,rot) = se3_ln_1(m,rot,
	theta = sqrt(rot*rot));
function se3_ln_1(m,rot,theta) = se3_ln_2(m,rot,theta,
	shtot = theta > 0.00001 ? sin(theta/2*180/PI)/theta : 0.5,
	halfrotator = so3_exp_rad(rot * -.5));
function se3_ln_2(m,rot,theta,shtot,halfrotator) =
concat( (halfrotator * translation_part(m) - 
	(theta > 0.001 
	? rot * ((translation_part(m) * rot) * (1-2*shtot) / (rot*rot))
	: rot * ((translation_part(m) * rot)/24)
	)) / (2 * shtot), rot);

//__se3_test = [20,-40,60,-80,100,-120];
//echo(UNITTEST_se3=norm(__se3_test-se3_ln(se3_exp(__se3_test))) < 1e-8);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// SO3

// so3

// use <linalg.scad>

function rodrigues_so3_exp(w, A, B) = [
[1.0 - B*(w[1]*w[1] + w[2]*w[2]), B*(w[0]*w[1]) - A*w[2],          B*(w[0]*w[2]) + A*w[1]],
[B*(w[0]*w[1]) + A*w[2],          1.0 - B*(w[0]*w[0] + w[2]*w[2]), B*(w[1]*w[2]) - A*w[0]],
[B*(w[0]*w[2]) - A*w[1],          B*(w[1]*w[2]) + A*w[0],          1.0 - B*(w[0]*w[0] + w[1]*w[1])]
];

function so3_exp(w) = so3_exp_rad(w/180*PI);
function so3_exp_rad(w) =
combine_so3_exp(w,
	w*w < 1e-8 
	? so3_exp_1(w*w)
	: w*w < 1e-6
	  ? so3_exp_2(w*w)
	  : so3_exp_3(w*w));

function combine_so3_exp(w,AB) = rodrigues_so3_exp(w,AB[0],AB[1]);

// Taylor series expansions close to 0
function so3_exp_1(theta_sq) = [
	1 - 1/6*theta_sq, 
	0.5
];

function so3_exp_2(theta_sq) = [
	1.0 - theta_sq * (1.0 - theta_sq/20) / 6,
	0.5 - 0.25/6 * theta_sq
];

function so3_exp_3_0(theta_deg, inv_theta) = [
	sin(theta_deg) * inv_theta,
	(1 - cos(theta_deg)) * (inv_theta * inv_theta)
];

function so3_exp_3(theta_sq) = so3_exp_3_0(sqrt(theta_sq)*180/PI, 1/sqrt(theta_sq));


function rot_axis_part(m) = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]*0.5;

function so3_ln(m) = 180/PI*so3_ln_rad(m);
function so3_ln_rad(m) = so3_ln_0(m,
	cos_angle = rot_cos_angle(m),
	preliminary_result = rot_axis_part(m));

function so3_ln_0(m, cos_angle, preliminary_result) = 
so3_ln_1(m, cos_angle, preliminary_result, 
	sin_angle_abs = sqrt(preliminary_result*preliminary_result));

function so3_ln_1(m, cos_angle, preliminary_result, sin_angle_abs) = 
	cos_angle > sqrt(1/2)
	? sin_angle_abs > 0
	  ? preliminary_result * asin(sin_angle_abs)*PI/180 / sin_angle_abs
	  : preliminary_result
	: cos_angle > -sqrt(1/2)
	  ? preliminary_result * acos(cos_angle)*PI/180 / sin_angle_abs
	  : so3_get_symmetric_part_rotation(
	      preliminary_result,
	      m,
	      angle = PI - asin(sin_angle_abs)*PI/180,
	      d0 = m[0][0] - cos_angle,
	      d1 = m[1][1] - cos_angle,
	      d2 = m[2][2] - cos_angle
			);

function so3_get_symmetric_part_rotation(preliminary_result, m, angle, d0, d1, d2) =
so3_get_symmetric_part_rotation_0(preliminary_result,angle,so3_largest_column(m, d0, d1, d2));

function so3_get_symmetric_part_rotation_0(preliminary_result, angle, c_max) =
	angle * unit(c_max * preliminary_result < 0 ? -c_max : c_max);

function so3_largest_column(m, d0, d1, d2) =
		d0*d0 > d1*d1 && d0*d0 > d2*d2
		?	[d0, (m[1][0]+m[0][1])/2, (m[0][2]+m[2][0])/2]
		: d1*d1 > d2*d2
		  ? [(m[1][0]+m[0][1])/2, d1, (m[2][1]+m[1][2])/2]
		  : [(m[0][2]+m[2][0])/2, (m[2][1]+m[1][2])/2, d2];

//__so3_test = [12,-125,110];
//echo(UNITTEST_so3=norm(__so3_test-so3_ln(so3_exp(__so3_test))) < 1e-8);