#include <stdio.h>
#include <math.h>

#define PI 3.1415927
#define g 9800					//定义重力加速度 mm/s^2 
//#define FRIC_K 0.0163			//空气阻力系数 
#define FRIC_K 0.0163//0.18
#define BALL_R 121 				//球的半径 mm
#define BALL_M 600 				//球的重量 g 
#define START_H 600				//出手高度 mm 
//#define GOAL_H 2430 + BALL_R 	//目标高度 mm
#define GOAL_H 2430+BALL_R
#define float double

//f = @(t) (Start(3)*Fric_K^2 + Vel(3)*Fric_K*m + g*m^2)/Fric_K^2 - (exp((-(Fric_K*t)/m))*(g*m^2 + Fric_K*Vel(3)*m))/Fric_K^2 - (g*m*t)/Fric_K-Target(3);
//df = @(t) (exp((-(Fric_K*t)/m))*(g*m^2 + Fric_K*Vel(3)*m))/(Fric_K*m) - (g*m)/Fric_K;
//hh = @(v0,t) (m*log((m + Fric_K*t*v0)/m))/Fric_K-target_y;

double f(double t, double vx, double vy)
{
	/*
	float term1 = (START_H * FRIC_K * FRIC_K + vy * FRIC_K * BALL_M + g * BALL_M * BALL_M) / (FRIC_K * FRIC_K);
	float term2 = (exp(-(FRIC_K * t) / BALL_M) * (g * BALL_M * BALL_M + FRIC_K * vy * BALL_M)) / (FRIC_K * FRIC_K);
	float term3 = (g * BALL_M * t) / FRIC_K;*/
	
	double term1 = (START_H + vy / FRIC_K * BALL_M + g * BALL_M / FRIC_K * BALL_M / FRIC_K);
	double term2 = (exp(-(FRIC_K * t) / BALL_M) * (g * BALL_M / FRIC_K * BALL_M / FRIC_K + vy / FRIC_K * BALL_M));
	double term3 = (g * BALL_M * t) / FRIC_K;
	return term1 - term2 - term3 - GOAL_H - 247;
	 
}

double df(double t, double vx, double vy)
{
	//return (exp((-(FRIC_K * t) / BALL_M)) * (g * BALL_M * BALL_M + FRIC_K * vy * BALL_M)) / (FRIC_K * BALL_M) - ((g * BALL_M) / FRIC_K);
	return (exp((-(FRIC_K * t) / BALL_M)) * (g * BALL_M / FRIC_K + vy)) - ((g * BALL_M) / FRIC_K);
}

double hh(double v0, double t, double dis)
{
	return (BALL_M * log((BALL_M + FRIC_K * t * v0) / BALL_M)) / FRIC_K - dis;
}

double NewtonIter(double vx, double vy)
{
	double Flytime = 0;	//飞行时间 
	double tol = 1e-6; 	//迭代最大误差 
	int max_iter = 20;	//最大迭代次数 
	bool flag = 0;		//是否获得正确飞行时间 
	for(double startx = 0; startx <= 5; startx += 0.5)
	{ 
	    double x0 = startx;
	    double x1; 
	    for(int iter = 1; iter <= max_iter; iter ++)
	    {
			x1 = x0 - f(x0, vx, vy) / df(x0, vx, vy);
			
			if(fabs(x1 - x0) < tol && df(x1, vx, vy) < 0)
			{
				flag = 1;
				Flytime = x1;
				break;
			}
			x0 = x1;
		}
	    if(flag)
	        break;
	}
	return Flytime;
} 

double BulletModelCalc(double angle, double dis)
{
	//二分法求解出射速度 
	double l_v0 = 0;
	double r_v0 = 20000;
	bool flag = 0;
	
	while(l_v0 <= r_v0)
	{
		double mid_v0 = (l_v0 + r_v0) / 2.0f;
		double mid_vx = mid_v0 * cosf(angle / 180.0f * PI);
    	double mid_vy = mid_v0 * sinf(angle / 180.0f * PI);
    	double mid_t = NewtonIter(mid_vx, mid_vy);
		double mid_now = hh(mid_vx, mid_t, dis);
		
		if(fabs(mid_now) < 1)
		{
			return mid_v0;
			flag = 1;
		}
		
		if(mid_now < 0)
			l_v0 = mid_v0 + 0.001;
		else
			r_v0 = mid_v0 - 0.001;
	}
	if(!flag)
		return 0;
}

int main()
{
	printf("%llf", BulletModelCalc(55,3175));
} 