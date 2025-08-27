%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 4: Constraints                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c, ceq] = CHAZEL_Constraints(x,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Positive Consumption Constraint & Positive Oil Constraint%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Compute consumption and oil resources based on x = [{Kt+1},{Rt+1},{pi0t},{pi2t}]:
%Step 1: Compute implied energy inputs
%Step 2: Compute carbon emissions and concentrations
%Step 3: Compute output and consumption

c = zeros(5*T+3,1);
%ceq = zeros(T-1,1); 

%%%%%%%%%%%%%%%%%%%%%%%%%
%%Step 1: Energy Inputs%%
%%%%%%%%%%%%%%%%%%%%%%%%%

oil = zeros(T,1);
    oil(1) = R0-x(T);
for i = 1:1:T-2;
    oil(1+i) = x(T+i-1)-x(T+i);
end
    ex_Oil = (x(T-1+T-2)-x(T-1+T-1))/(x(T-1+T-2));    %Fraction of oil left extracted in period T-1
    oil(T) = x(T-1+T-1)*ex_Oil;
 for i = 1:1:T;
     c(T+i) = (-1)*(oil(i)-0.0001);                   %Positive oil constraint
 end
    c(2*T+1) = (ex_Oil-1);
coal = zeros(T,1);
for i = 1:1:T;
    coal(i) = x(2*(T-1)+i)*(A2t(i)*N);
end
%wind = zeros(T,1);
%for i = 1:1:T;
    %wind(i) = x(2*(T-1)+T+i)*(A3t(i)*N);
%end 

%%%%%%%% SELF ADDED %%%%%%%%%%%%%%%%%
%%Primary mineral extraction, eq(12)
min_p = zeros(T,1);
for i = 1:1:T;
    min_p(i) = x(2*(T-1)+2*T+i)*(Ap(i)*N);
end

%%Secondary mineral extraction, eq(15)
min_s = zeros(T,1);
for i = 1:1:T;
    min_s(i) = x(2*(T-1)+3*T+i)*(As(i)*N);
end

%%Green capital production, eq (11) 
green = zeros(T,1);
for i = 1:1:T;
    green(i) = ((kappaS*min_s(i)^rho_green)+(kappaP*min_p(i)^rho_green))^(1/(rho_green));
end 

%% Primary mineral stock 
Mp = zeros(T,1);
Mp(1) = M0;
for i = 1:1:T-2;
    Mp(i+1) = Mp(i) - min_p(i);
end
    ex_Mp = (Mp(28)-Mp(29))/(Mp(28));    %Fraction of Mp left extracted in final period
    Mp(T) = Mp(29)*ex_Mp;
 for i = 1:1:T;
     c(2*T+1+i) = (-1)*(min_p(i)-0.0001);                   %Positive primary mineral constraint
 end
     c(3*T+1) = (ex_Mp-1);

%% Secondary mineral stock
Ms = zeros(T,1);
Ms(1) = Ms0; % Stock at end of period 1
for i = 1:1:T-2;
    Ms(i+1) = Ms(i) + min_p(i);
end
    ex_Ms = (Ms(28)-Ms(29))/(Ms(28));    %Fraction of Ms left extracted in final period
    Ms(T) = Ms(29)*ex_Ms;
 for i = 1:1:T;
     c(3*T+2+i) = (-1)*(min_s(i)-0.0001);                   %Positive secondary mineral constraint
 end
     c(4*T+1) = (ex_Ms-1);

%%Low carbon energy production, eq (10)
E3 = zeros(T,1);
for i = 1:1:T;
    E3(i) = (kappaL*(x(2*(T-1)+T+i)*A3t(i)*N)^rho_E3+kappaG*(psi*green(i))^rho_E3)^(1/rho_E3);
end


%%Same as GHKT and Chazel: 
energy = zeros(T,1);
for i = 1:1:T; 
    %energy(i) = ((kappa1*oil(i)^rho)+(kappa2*coal(i)^rho)+(kappa3*wind(i)^rho))^(1/rho);
    energy(i) = ((kappa1*oil(i)^rho)+(kappa2*coal(i)^rho)+(kappa3*E3(i)^rho))^(1/rho);
end


%%%%%%%% New Constraints %%%%%%%%

%% Positive primary mineral stock Mp
%for i = 1:1:T;
    %c(2*T+1+i) = (-1)*(Mp(i)-0.0001);
%end
%Positive secondary mineral stock Ms
%for i = 1:1:T;
    %c(4*T+3+i) = (-1)*(Ms(i)-0.0001);
%end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Step 2: Carbon Emissions and Concentrations%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

emiss = zeros(T,1);
for i = 1:1:T;
    emiss(i) = oil(i)+ypsilon(i)*coal(i);
end

S1t = zeros(T,1);        %Non-depreciating stock
S2t_Sbar = zeros(T,1);   %Depreciating stock (S2t-Sbar)
St = zeros(T,1);         %Total carbon concentrations

S1t(1) = S1_2000+phiL*emiss(1);
S2t_Sbar(1) = (1-phi)*(S2_2000-Sbar)+phi0*(1-phiL)*emiss(1);
St(1) = Sbar+S1t(1)+S2t_Sbar(1);

for i = 1:1:T-1;
    S1t(1+i) = S1t(i)+phiL*emiss(1+i);
    S2t_Sbar(1+i) = (1-phi)*S2t_Sbar(i)+phi0*(1-phiL)*emiss(1+i);
    St(1+i) = Sbar+S1t(1+i)+S2t_Sbar(1+i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Step 3: Output and Consumption%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Yt = zeros(T,1);
Ct = zeros(T,1);
Kt1 = zeros(T,1);
%Yt(1) = At(1)*(exp((-gamma(1))*(St(1)-Sbar)))*(K0^alpha)*(((1-x(2*(T-1)+1)-x(2*(T-1)+T+1))*N)^(1-alpha-v))*(energy(1)^v);
Yt(1) = At(1)*(exp((-gamma(1))*(St(1)-Sbar)))*(K0^alpha)*(((1-x(2*(T-1)+1)-x(2*(T-1)+T+1)-x(2*(T-1)+2*T+1)-x(2*(T-1)+3*T+1))*N)^(1-alpha-v))*(energy(1)^v);
Ct(1) = (1-x(1))*Yt(1);
Kt1(1) = x(1)*Yt(1)+(1-Delta)*K0;
for i = 1:1:T-2;
    %Yt(1+i) = At(1+i)*(exp((-gamma(1+i))*(St(1+i)-Sbar)))*(Kt1(i)^alpha)*(((1-x(2*(T-1)+1+i)-x(2*(T-1)+T+1+i))*N)^(1-alpha-v))*(energy(1+i)^v);
    Yt(1+i) = At(1+i)*(exp((-gamma(1+i))*(St(1+i)-Sbar)))*(Kt1(i)^alpha)*(((1-x(2*(T-1)+1+i)-x(2*(T-1)+T+1+i)-x(2*(T-1)+2*T+i)-x(2*(T-1)+3*T+i))*N)^(1-alpha-v))*(energy(1+i)^v);
    Kt1(1+i) = x(1+i)*Yt(1+i)+(1-Delta)*Kt1(i);
    Ct(1+i) = (1-x(i+1))*Yt(1+i); 
end
%Yt(T) = At(T)*(exp((-gamma(T))*(St(T)-Sbar)))*(Kt1(T-1)^alpha)*(((1-x(2*(T-1)+T)-x(2*(T-1)+2*T))*N)^(1-alpha-v))*(energy(T)^v);
Yt(T) = At(T)*(exp((-gamma(T))*(St(T)-Sbar)))*(Kt1(T-1)^alpha)*(((1-x(2*(T-1)+T)-x(2*(T-1)+2*T)-x(2*(T-1)+3*T)-x(2*(T-1)+4*T))*N)^(1-alpha-v))*(energy(T)^v);
theta = x(T-1);
Ct(T) = Yt(T)*(1-theta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Positive Consumption Constraint:%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:1:T;
    c(i) = (-1)*(Ct(i)-0.0001);
end


%%%Equality Constraint for Benchmark Case%%%
ceq = [];

end