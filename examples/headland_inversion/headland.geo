basin_x = 20000;
basin_y = 6000;

headland_x_scale = 0.2;
headland_y = 2000;

site_x = 1600;
site_y = 1000;
site_x_start = basin_x/2-site_x/2;
site_x_end = basin_x/2+site_x/2;

site_y_start = basin_y/2 - 500;
site_y_end = site_y_start+site_y;

element_size = 150;
element_size_coarse = 400;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};


// Generate nodes for the headland
res = 100;
For k In {0:res:1}
    x = basin_x/res*k;
    b = 100;
    y = basin_y - headland_y*Exp(-0.5*((headland_x_scale*(x-basin_x/2))/b)^2);
	Point(10+k) = {x, y, 0, element_size_coarse};
EndFor

// Generate lines for the headland

BSpline(100) = { 10 : res+10 };

Line(101) = {10, 1};
Line(102) = {1, 2};
Line(103) = {2, res+10};
Line Loop(104) = {100, -103, -102, -101};

// Generate site nodes
Point(1000) = {site_x_start, site_y_start, 0, element_size};
Extrude{site_x, 0, 0} { Point{1000}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{105}; Layers{site_y/element_size}; }
Line Loop(110) = {106, -108, -105, 107};
Plane Surface(111) = {104, 110};
Plane Surface(112) = {110};
Physical Line(1) = {101};
Physical Line(2) = {103};
Physical Line(3) = {100, 102};
Physical Surface(1) = {111};
Physical Surface(2) = {112};
