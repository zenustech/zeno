vec3 perlin_hash22(vec3 p)
{
    p = vec3( dot(p,vec3(127.1,311.7,284.4)),
              dot(p,vec3(269.5,183.3,162.2)),
	      	  dot(p,vec3(228.3,164.9,126.0)));

    return -1.0 + 2.0 * fract(sin(p)*43758.5453123);
}

float perlin_lev1(vec3 p)
{
    vec3 pi = floor(p);
    vec3 pf = p - pi;
    vec3 w = pf * pf * (3.0 - 2.0 * pf);
    return .08 + .8 + (mix(
			    mix(
				    mix(
					    dot(perlin_hash22(pi + vec3(0, 0, 0)), pf - vec3(0, 0, 0)),
					    dot(perlin_hash22(pi + vec3(1, 0, 0)), pf - vec3(1, 0, 0)),
					    w.x),
				    mix(
					    dot(perlin_hash22(pi + vec3(0, 1, 0)), pf - vec3(0, 1, 0)),
					    dot(perlin_hash22(pi + vec3(1, 1, 0)), pf - vec3(1, 1, 0)),
					    w.x),
				    w.y),
			    mix(
				    mix(
					    dot(perlin_hash22(pi + vec3(0, 0, 1)), pf - vec3(0, 0, 1)),
					    dot(perlin_hash22(pi + vec3(1, 0, 1)), pf - vec3(1, 0, 1)),
					    w.x),
				    mix(
					    dot(perlin_hash22(pi + vec3(0, 1, 1)), pf - vec3(0, 1, 1)),
					    dot(perlin_hash22(pi + vec3(1, 1, 1)), pf - vec3(1, 1, 1)),
					    w.x),
				    w.y),
			    w.z));
}

float perlin(float p,int n,vec3 a)
{
    float total = 0;
    for(int i=0; i<n; i++)
    {
        float frequency = pow(2,i);
        float amplitude = pow(p,i);
        total = total + perlin_lev1(a * frequency) * amplitude;
    }

    return total;
}
