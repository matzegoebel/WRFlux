!WRF:MODEL_LAYER:DYNAMICS
!
MODULE module_avgflx_em

  USE module_bc
  USE module_model_constants
  USE module_wrf_error
  USE module_state_description

CONTAINS

!-------------------------------------------------------------------------------


  subroutine zero_avgflx(avgflx_rum,avgflx_rvm,avgflx_wwm, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu, do_avgflx_em, output_avgfluxes,    &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1, &
       & hist_mean, hist_mean_8w)  
    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte
    INTEGER , INTENT(IN)        :: do_avgflx_em, output_avgfluxes
    LOGICAL, INTENT(IN) :: do_cu

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL , OPTIONAL, DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean), INTENT(INOUT) :: hist_mean
    REAL , OPTIONAL, DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean_8w), INTENT(INOUT) :: hist_mean_8w

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    INTEGER :: i,j,k

    if (output_avgfluxes .eq. 1) then
      DO j=jts,jte
         DO k=kts,kte
            DO i=its,ite
               hist_mean(i,k,j,:) = 0.
               hist_mean_8w(i,k,j,:) = 0.
            end DO
         end DO
       end DO
    endif

    if (do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = 0.
             avgflx_rvm(i,k,j) = 0.
             avgflx_wwm(i,k,j) = 0.
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = 0.
                avgflx_cfd1(i,k,j) = 0.
                avgflx_dfu1(i,k,j) = 0.
                avgflx_efu1(i,k,j) = 0.
                avgflx_dfd1(i,k,j) = 0.
                avgflx_efd1(i,k,j) = 0.
             end DO
          end DO
       end DO
    end if

    return
  end subroutine zero_avgflx

  subroutine upd_avgflx(avgflx_count,avgflx_rum,avgflx_rvm,avgflx_wwm, &
       &   ru_m, rv_m, ww_m, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu,    &
       & cfu1,cfd1,dfu1,efu1,dfd1,efd1,          &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1,   &
       & config_flags,                                                              &
       & sgs_fluxes, sgs_fluxes_8w, nba_mij, hist_mean, hist_mean_8w,               &
       & ruth_tot, rvth_tot, wwth_tot, ruq_tot, rvq_tot, wwq_tot,                   &
       & ruu_tot, rvu_tot, wwu_tot, ruv_tot, rvv_tot, wwv_tot, ruw_tot, rvw_tot, www_tot, &
       & t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp,                         &
       & U, V, W, ru, rv, ww, thm, Q,                                               &
       & ph_1, ph_2, phb, zx, zy,                                                   &
       & dnw, fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3, cfn, cfn1,          &
       & last_timestep, dt                                                          )
    

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    INTEGER , INTENT(IN)        :: avgflx_count
    LOGICAL, INTENT(IN) :: do_cu, last_timestep
    REAL, INTENT(IN) :: dt, cf1, cf2, cf3, cfn, cfn1

    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
         ru_m, rv_m, ww_m,             & 
         U, V, W, ru, rv, ww,  &
         thm, Q, ph_1, ph_2, phb, zx, zy,  &
         ruth_tot, rvth_tot, wwth_tot,     &
         ruq_tot, rvq_tot, wwq_tot,        &
         ruu_tot, rvu_tot, wwu_tot,        &
         ruv_tot, rvv_tot, wwv_tot,        &
         ruw_tot, rvw_tot, www_tot,        &
         t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp

    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_sgs_fluxes), INTENT(IN) :: sgs_fluxes
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_sgs_fluxes_8w), INTENT(IN) :: sgs_fluxes_8w
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_nba_mij), INTENT(IN) :: nba_mij
    REAL , DIMENSION( kms:kme ) ,    INTENT(IN   ) :: dnw, fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), INTENT(IN   ) :: mut



    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags


    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean), INTENT(INOUT) :: hist_mean
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean_8w), INTENT(INOUT) :: hist_mean_8w


    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(IN) ::    &
         cfu1,cfd1,dfu1,efu1,dfd1,efd1
    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    REAL,    DIMENSION( its-1:ite , kts:kte , jts-1:jte ) ::    rho_d, th, thd

    INTEGER :: i,j,k
    REAL :: local_count, rho_8w, rhox, rhoy, rhoz, rho_mean, rhox_mean, rhoy_mean, rhoz_mean, &
            thetax, thetay, thetaz, qx, qy, qz, ux, uy, uz, vx, vy, vz, wx, wy, wz, qfac,     &
            zx_8w, zy_8w, u_8w, v_8w, dzdt, &
            corr_uth, corr_vth, corr_dthdt, &
            corr_uq, corr_vq, corr_dqdt,    &
            corr_uu, corr_vu, corr_dudt, &
            corr_uv, corr_vv, corr_dvdt, &
            corr_uw, corr_vw, corr_dwdt, &
            wdiag, muz, mu, dphidn

    local_count = real(avgflx_count)

    if (config_flags%output_avgfluxes .eq. 1) then

    DO j=jts-1,jte
    DO k=kts,kte
    DO i=its-1,ite
      !calculate dry density
      mu = c1h(k)*mut(i,j)+c2h(k)
      dphidn = ((phb(i,k+1,j)+ph_2(i,k+1,j)) - (phb(i,k,j)+ph_2(i,k,j)))/dnw(k)
      rho_d(i,k,j) = mu/(-dphidn)
      hist_mean(i,k,j,P_RHOD_MEAN) = (local_count*hist_mean(i,k,j,P_RHOD_MEAN)  + rho_d(i,k,j)  )/(local_count+1.)
      IF ( config_flags%use_theta_m .EQ. 1 ) THEN
        thd(i,k,j) = (thm(i,k,j) + t0)/(1 + rvovrd*Q(i,k,j)) - t0
        IF ( config_flags%output_dry_theta_fluxes .eq. 1) THEN
          th(i,k,j) = thd(i,k,j)
        ELSE
          th(i,k,j) = thm(i,k,j)
        ENDIF
      ELSE
        thd(i,k,j) = thm(i,k,j)
        th(i,k,j) = thm(i,k,j)
      ENDIF
    ENDDO
    ENDDO
    ENDDO

    DO j=jts,jte
      DO k=kts,kde
        DO i=its,ite

          hist_mean_8w(i,k,j,P_Z_MEAN) = (local_count*hist_mean_8w(i,k,j,P_Z_MEAN)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/(local_count+1.)

          IF (config_flags%output_res_fluxes==1) THEN
            !stagger density
            call stagger_z(rho_8w, rho_d(i,:,j),               &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )

            if (config_flags%hesselberg_avg==1) then
              rhox = 0.5*(rho_d(i,k,j)+rho_d(i-1,k,j))
              rhoy = 0.5*(rho_d(i,k,j)+rho_d(i,k,j-1))
              rhoz = rho_8w
            else
              rhox = 1
              rhoy = 1
              rhoz = 1
            endif


            !determine staggered variables from total fluxes

            call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                           ruth_tot, rvth_tot, wwth_tot,&
                                           ru, rv, ww, th,              &
                                           cf1, cf2, cf3, cfn, cfn1,    &
                                           i, j, k, kde,                &
                                           ims, ime, jms, jme, kms, kme  )

            !staggered q, different vertical velocity needed
            call get_staggered_from_fluxes(qx, qy, qz,                 &
                                           ruq_tot, rvq_tot, wwq_tot,  &
                                           ru_m, rv_m, ww_m, Q,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )
    !TODO: which velocity? and staggering?
            call get_staggered_from_fluxes(ux, uy, uz,                 &
                                           ruu_tot, rvu_tot, wwu_tot,  &
                                           ru_m, rv_m, ww_m, U,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )
            call get_staggered_from_fluxes(vx, vy, vz,                 &
                                           ruv_tot, rvv_tot, wwv_tot,  &
                                           ru_m, rv_m, ww_m, V,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )

           !TODO different staggering for w!!
            call get_staggered_from_fluxes(wx, wy, wz,                 &
                                           ruw_tot, rvw_tot, www_tot,  &
                                           ru_m, rv_m, ww_m, W,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )

            !compute cartesian corrections
            zx_8w = 0.5*(zx(i,k,j)+zx(i+1,k,j))
            zy_8w = 0.5*(zy(i,k,j)+zy(i,k,j+1))
            dzdt = (ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)
            call stagger_z(u_8w, U(i,:,j)+U(i+1,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )
            call stagger_z(v_8w, V(i,:,j)+V(i,:,j+1),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )

            wdiag = dzdt + zx_8w*u_8w + zy_8w*v_8w + ww(i,k,j)/(-rho_8w*g)
            muz = c1f(k)*mut(i,j)+c2f(k)

            corr_uth = rho_8w*zx_8w*u_8w*thetaz
            corr_vth = rho_8w*zy_8w*v_8w*thetaz
            corr_dthdt = rho_8w*dzdt*thetaz

            corr_uq = rho_8w*zx_8w*u_8w*qz
            corr_vq = rho_8w*zy_8w*v_8w*qz
            corr_dqdt = rho_8w*dzdt*qz

            corr_uu = rho_8w*zx_8w*u_8w*uz
            corr_vu = rho_8w*zy_8w*v_8w*uz
            corr_dudt = rho_8w*dzdt*uz

            corr_uv = rho_8w*zx_8w*u_8w*vz
            corr_vv = rho_8w*zy_8w*v_8w*vz
            corr_dvdt = rho_8w*dzdt*vz

            corr_uw = rho_8w*zx_8w*u_8w*wz
            corr_vw = rho_8w*zy_8w*v_8w*wz
            corr_dwdt = rho_8w*dzdt*wz


            if (last_timestep .and. config_flags%hesselberg_avg==1) then
              !normalize means by density
              rho_mean = hist_mean(i,k,j,P_RHOD_MEAN)
              rhox_mean = 0.5*(hist_mean(i,k,j,P_RHOD_MEAN) + hist_mean(i-1,k,j,P_RHOD_MEAN))
              rhoy_mean = 0.5*(hist_mean(i,k,j,P_RHOD_MEAN) + hist_mean(i,k,j-1,P_RHOD_MEAN))

              call stagger_z(rhoz_mean, hist_mean(i,:,j,P_RHOD_MEAN), &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                             i, j, k, kms, kme, kde                   )
            ELSE
              rho_mean  = 1
              rhox_mean = 1
              rhoy_mean = 1
              rhoz_mean = 1
            ENDIF

            !mean variables
            hist_mean(i,k,j,P_U_MEAN) = (local_count*hist_mean(i,k,j,P_U_MEAN) + U(i,k,j)*rhox  )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_V_MEAN) = (local_count*hist_mean(i,k,j,P_V_MEAN) + V(i,k,j)*rhoy  )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_W_MEAN) = (local_count*hist_mean_8w(i,k,j,P_W_MEAN) + W(i,k,j)*rhoz  )/(local_count+1.)/rhoz_mean
            hist_mean_8w(i,k,j,P_OM_MEAN) = (local_count*hist_mean_8w(i,k,j,P_OM_MEAN) + ww(i,k,j)/muz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean_8w(i,k,j,P_WD_MEAN) = (local_count*hist_mean_8w(i,k,j,P_WD_MEAN) + wdiag*rhoz )/(local_count+1.)/rhoz_mean

            hist_mean(i,k,j,P_TH_MEAN) = (local_count*hist_mean(i,k,j,P_TH_MEAN) + thd(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_THX_MEAN) = (local_count*hist_mean(i,k,j,P_THX_MEAN) + thetax*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_THY_MEAN) = (local_count*hist_mean(i,k,j,P_THY_MEAN) + thetay*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_THZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_THZ_MEAN) + thetaz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean(i,k,j,P_THTH_MEAN) = (local_count*hist_mean(i,k,j,P_THTH_MEAN) + thd(i,k,j)**2 )/(local_count+1.)

            hist_mean(i,k,j,P_Q_MEAN) = (local_count*hist_mean(i,k,j,P_Q_MEAN) + Q(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_QX_MEAN) = (local_count*hist_mean(i,k,j,P_QX_MEAN) + qx*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_QY_MEAN) = (local_count*hist_mean(i,k,j,P_QY_MEAN) + qy*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_QZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_QZ_MEAN) + qz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean(i,k,j,P_QQ_MEAN) = (local_count*hist_mean(i,k,j,P_QQ_MEAN) + Q(i,k,j)*2 )/(local_count+1.)

            hist_mean(i,k,j,P_U_MEAN) = (local_count*hist_mean(i,k,j,P_U_MEAN) + U(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_UX_MEAN) = (local_count*hist_mean(i,k,j,P_UX_MEAN) + ux*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_UY_MEAN) = (local_count*hist_mean(i,k,j,P_UY_MEAN) + uy*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_UZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_UZ_MEAN) + uz*rhoz )/(local_count+1.)/rhoz_mean

            hist_mean(i,k,j,P_V_MEAN) = (local_count*hist_mean(i,k,j,P_V_MEAN) + V(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_VX_MEAN) = (local_count*hist_mean(i,k,j,P_VX_MEAN) + vx*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_VY_MEAN) = (local_count*hist_mean(i,k,j,P_VY_MEAN) + vy*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_VZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_VZ_MEAN) + vz*rhoz )/(local_count+1.)/rhoz_mean

            hist_mean(i,k,j,P_W_MEAN) = (local_count*hist_mean(i,k,j,P_W_MEAN) + W(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_WX_MEAN) = (local_count*hist_mean(i,k,j,P_WX_MEAN) + wx*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_WY_MEAN) = (local_count*hist_mean(i,k,j,P_WY_MEAN) + wy*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_WZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_WZ_MEAN) + wz*rhoz )/(local_count+1.)/rhoz_mean

            !mean total fluxes
            !TODO: momentum fluxes from advection module

            hist_mean(i,k,j,P_RUTH_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUTH_TOT_MEAN) + ruth_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVTH_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVTH_TOT_MEAN) + rvth_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWTH_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWTH_TOT_MEAN) + wwth_tot(i,k,j) )/(local_count+1.)

            hist_mean(i,k,j,P_RUQ_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUQ_TOT_MEAN) + ruq_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVQ_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVQ_TOT_MEAN) + rvq_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWQ_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWQ_TOT_MEAN) + wwq_tot(i,k,j) )/(local_count+1.)

            hist_mean(i,k,j,P_RUU_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUU_TOT_MEAN) + ruu_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVU_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVU_TOT_MEAN) + rvu_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWU_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWU_TOT_MEAN) + wwu_tot(i,k,j) )/(local_count+1.)

            hist_mean(i,k,j,P_RUV_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUV_TOT_MEAN) + ruv_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVV_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVV_TOT_MEAN) + rvv_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWV_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWV_TOT_MEAN) + wwv_tot(i,k,j) )/(local_count+1.)

            hist_mean(i,k,j,P_RUW_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUW_TOT_MEAN) + ruw_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVW_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVW_TOT_MEAN) + rvw_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWW_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWW_TOT_MEAN) + www_tot(i,k,j) )/(local_count+1.)

            !mean correction terms
            hist_mean_8w(i,k,j,P_CORR_UTH)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UTH) + corr_uth )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VTH)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VTH) + corr_vth )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DTHDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DTHDT) + corr_dthdt )/(local_count+1.)

            hist_mean_8w(i,k,j,P_CORR_UQ)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UQ) + corr_uq )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VQ)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VQ) + corr_vq )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DQDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DQDT) + corr_dqdt )/(local_count+1.)

            hist_mean_8w(i,k,j,P_CORR_UU)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UU) + corr_uu )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VU)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VU) + corr_vu )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DUDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DUDT) + corr_dudt )/(local_count+1.)

            hist_mean_8w(i,k,j,P_CORR_UV)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UV) + corr_uv )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VV)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VV) + corr_vv )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DVDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DVDT) + corr_dvdt )/(local_count+1.)

            hist_mean_8w(i,k,j,P_CORR_UW)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UW) + corr_uw )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VW)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VW) + corr_vw )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DWDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DWDT) + corr_dwdt )/(local_count+1.)

          ENDIF

          hist_mean(i,k,j,P_T_TEND_RADLW_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_RADLW_MEAN) + t_tend_radlw(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_T_TEND_RADSW_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_RADSW_MEAN) + t_tend_radsw(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_T_TEND_MP_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_MP_MEAN) + t_tend_mp(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_Q_TEND_MP_MEAN)  = (local_count*hist_mean(i,k,j,P_Q_TEND_MP_MEAN) + qv_tend_mp(i,k,j) )/(local_count+1.)
          !TODO: temperature tendencies from cumulus scheme; small time step tendencies from momentum
          IF (config_flags%output_sgs_fluxes==1) THEN
            hist_mean(i,k,j,P_sgs_UU_mean)  = (local_count*hist_mean(i,k,j,P_sgs_UU_mean)  + sgs_fluxes(i,k,j,P_sgs_UU) )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VV_mean)  = (local_count*hist_mean(i,k,j,P_sgs_VV_mean)  + sgs_fluxes(i,k,j,P_sgs_VV)  )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_WW_mean)  = (local_count*hist_mean(i,k,j,P_sgs_WW_mean)  + sgs_fluxes(i,k,j,P_sgs_WW)  )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_UV_mean)  = (local_count*hist_mean(i,k,j,P_sgs_UV_mean)  + sgs_fluxes(i,k,j,P_sgs_UV) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_UW_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_UW_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_UW) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_VW_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_VW_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_VW) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WU_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_WU_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_WU) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WV_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_WV_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_WV) )  / (local_count+1.)

            hist_mean(i,k,j,P_sgs_UTH_mean) = (local_count*hist_mean(i,k,j,P_sgs_UTH_mean) + sgs_fluxes(i,k,j,P_sgs_UTH) ) / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VTH_mean) = (local_count*hist_mean(i,k,j,P_sgs_VTH_mean) + sgs_fluxes(i,k,j,P_sgs_VTH) ) / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WTH_mean) = (local_count*hist_mean_8w(i,k,j,P_sgs_WTH_mean) + sgs_fluxes_8w(i,k,j,P_sgs_WTH) ) / (local_count+1.)

            hist_mean(i,k,j,P_sgs_UQ_mean) = (local_count*hist_mean(i,k,j,P_sgs_UQ_mean) + sgs_fluxes(i,k,j,P_sgs_UQ) ) / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VQ_mean) = (local_count*hist_mean(i,k,j,P_sgs_VQ_mean) + sgs_fluxes(i,k,j,P_sgs_VQ) ) / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WQ_mean) = (local_count*hist_mean_8w(i,k,j,P_sgs_WQ_mean) + sgs_fluxes_8w(i,k,j,P_sgs_WQ) ) / (local_count+1.)
          ENDIF

        end DO
      end DO
    end DO
    endif

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = (local_count*avgflx_rum(i,k,j) + ru_m(i,k,j))/(local_count+1.)
             avgflx_rvm(i,k,j) = (local_count*avgflx_rvm(i,k,j) + rv_m(i,k,j))/(local_count+1.)
             avgflx_wwm(i,k,j) = (local_count*avgflx_wwm(i,k,j) + ww_m(i,k,j))/(local_count+1.)
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) &
         & .and. present(cfu1) .and. present(cfd1) .and. present(dfu1) &
         & .and. present(efu1) .and. present(dfd1) .and. present(efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = (local_count*avgflx_cfu1(i,k,j) + &
                     & cfu1(i,k,j)) / (local_count+1.)
                avgflx_cfd1(i,k,j) = (local_count*avgflx_cfd1(i,k,j) + &
                     & cfd1(i,k,j)) / (local_count+1.)
                avgflx_dfu1(i,k,j) = (local_count*avgflx_dfu1(i,k,j) + &
                     & dfu1(i,k,j)) / (local_count+1.)
                avgflx_efu1(i,k,j) = (local_count*avgflx_efu1(i,k,j) + &
                     & efu1(i,k,j)) / (local_count+1.)
                avgflx_dfd1(i,k,j) = (local_count*avgflx_dfd1(i,k,j) + &
                     & dfd1(i,k,j)) / (local_count+1.)
                avgflx_efd1(i,k,j) = (local_count*avgflx_efd1(i,k,j) + &
                     & efd1(i,k,j)) / (local_count+1.)
             end DO
          end DO
       end DO
    end if

    return
  end subroutine upd_avgflx

  subroutine stagger_z(var_stag, var,                      &
                       cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                       i, j, k, kms, kme, kde              )

    REAL, INTENT(OUT) :: var_stag
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  i, j, k, kms, kme, kde
    REAL, DIMENSION(kms:kme) , INTENT(IN) :: var, fnm, fnp

    if ( k .eq. kde) then
      var_stag = cfn*var(kde-1)+cfn1*var(kde-2)
    elseif ( k .eq. 1) then
      var_stag = cf1*var(1)+cf2*var(2)+cf3*var(3)
    else
      var_stag = fnm(k)*var(k)+ fnp(k)*var(k-1)
    endif

  end subroutine stagger_z

  subroutine get_staggered_from_fluxes(varx, vary, varz,                &
                                       ruvar_tot, rvvar_tot, wwvar_tot, &
                                       ru, rv, ww, var,                 &
                                       cf1, cf2, cf3, cfn, cfn1,        &
                                       i, j, k, kde,                    &
                                       ims, ime, jms, jme, kms, kme     )

    IMPLICIT NONE

    REAL, INTENT(OUT) :: varx, vary, varz
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  i, j, k, kde,             &
                            ims, ime, jms, jme, kms, kme
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
                        ruvar_tot, rvvar_tot, wwvar_tot, &
                        ru, rv, ww, var

    varx = ruvar_tot(i,k,j)/ru(i,k,j)
    vary = rvvar_tot(i,k,j)/rv(i,k,j)
    if ( k .eq. kde) then
      varz = cfn*var(i,kde-1,j)+cfn1*var(i,kde-2,j)
    elseif ( k .eq. 1) then
      varz = cf1*var(i,1,j)+cf2*var(i,2,j)+cf3*var(i,3,j)
    else
      varz = wwvar_tot(i,k,j)/ww(i,k,j)
    endif
  end subroutine get_staggered_from_fluxes

end MODULE module_avgflx_em
