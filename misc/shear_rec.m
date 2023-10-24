function y=shear_rec(shear1,shear2)
N_grid=size(shear1,1);
[theta_x,theta_y]=meshgrid(linspace(-N_grid+1,N_grid-1,2*N_grid-1));
D_starkernel=-1./(theta_x+1j*theta_y).^2;
D_starkernel(N_grid,N_grid)=0;
y=real(ifft2(fft2(D_starkernel,3*N_grid-2,3*N_grid-2).*fft2(shear1+1j*shear2,3*N_grid-2,3*N_grid-2)))/pi;
y=y(N_grid:2*N_grid-1,N_grid:2*N_grid-1);
