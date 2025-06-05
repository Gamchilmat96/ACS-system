import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm # 영어만 사용하므로 한글 폰트 매니저는 불필요

def analyze_bullet_data(file_path):
    """
    CSV 파일을 읽어 pitch별 distance를 분석하고 시각화하며,
    pitch를 distance의 함수로 나타내는 방정식을 구합니다.
    """
    try:
        df_bullets = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}") # 오류 메시지는 영어로 유지
        return
    except Exception as e:
        print(f"Error loading file: {e}") # 오류 메시지는 영어로 유지
        return

    print("--- Data Info ---") # 프로그램 출력은 영어로 유지
    df_bullets.info()
    print("\n--- First 5 rows of Data ---") # 프로그램 출력은 영어로 유지
    print(df_bullets.head())

    # 1. 데이터 집계: 각 pitch 값에 대한 평균 distance 계산
    df_aggregated = df_bullets.groupby('pitch')['distance'].mean().reset_index()
    df_aggregated = df_aggregated.sort_values(by='pitch') # pitch 기준으로 정렬

    print("\n--- Mean Distance per Pitch (Sample) ---") # 프로그램 출력은 영어로 유지
    print(df_aggregated.head())

    # 2. 시각화: Pitch에 따른 평균 Distance
    plt.figure(figsize=(12, 7))
    plt.scatter(df_aggregated['pitch'], df_aggregated['distance'], color='blue', label='Mean Distance per Pitch (Data)')
    plt.title('Mean Projectile Distance vs. Pitch') # 그래프 제목 (영어)
    plt.xlabel('Pitch (degrees)')                   # X축 레이블 (영어)
    plt.ylabel('Mean Distance')                     # Y축 레이블 (영어)
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. 방정식 구하기: pitch = g(distance)
    # X축을 distance, Y축을 pitch로 하여 모델 피팅
    x_data_for_pitch_model = df_aggregated['distance']
    y_data_for_pitch_model = df_aggregated['pitch']

    # 3차 다항식으로 피팅 (pitch = c0*distance^3 + c1*distance^2 + c2*distance + c3)
    pitch_model_degree = 3
    coeffs_pitch_model = np.polyfit(x_data_for_pitch_model, y_data_for_pitch_model, pitch_model_degree)
    poly_func_pitch_model = np.poly1d(coeffs_pitch_model)

    print("\n--- 'pitch = g(distance)' Model ---") # 프로그램 출력은 영어로 유지
    print(f"Coefficients for {pitch_model_degree}rd degree polynomial (highest power first c0, c1, c2, c3):") # 프로그램 출력은 영어로 유지
    coeff_labels = [f'c{i}' for i in range(pitch_model_degree + 1)]
    for label, coeff in zip(coeff_labels, coeffs_pitch_model):
        print(f"  {label}: {coeff:.8f}") # 계수 출력 (영어)
    
    c0, c1, c2, c3 = coeffs_pitch_model
    print(f"\nEquation: pitch ≈ {c0:.8f}*distance³ + {c1:.8f}*distance² + {c2:.8f}*distance + {c3:.8f}") # 방정식 표현 (영어)


    # 4. 'pitch = g(distance)' 모델 시각화
    plt.figure(figsize=(12, 7))
    plt.scatter(x_data_for_pitch_model, y_data_for_pitch_model, color='green', label='Data (Distance, Pitch)')

    # 부드러운 곡선을 위해 더 많은 포인트 생성하여 플롯
    distance_range_for_plot = np.linspace(x_data_for_pitch_model.min(), x_data_for_pitch_model.max(), 200)
    pitch_predicted_values = poly_func_pitch_model(distance_range_for_plot)

    plt.plot(distance_range_for_plot, pitch_predicted_values, color='purple', 
             label=f'Polynomial Fit (degree {pitch_model_degree}, pitch = g(distance))')
    plt.title('Pitch vs. Mean Projectile Distance (pitch = g(distance) Model)') # 그래프 제목 (영어)
    plt.xlabel('Mean Distance')                   # X축 레이블 (영어)
    plt.ylabel('Pitch (degrees)')                 # Y축 레이블 (영어)
    plt.legend()
    plt.grid(True)
    plt.show()

    # (참고용) distance = f(pitch) 모델도 생성 및 시각화
    print("\n--- (Reference) 'distance = f(pitch)' Model ---") # 프로그램 출력은 영어로 유지
    coeffs_dist_model = np.polyfit(df_aggregated['pitch'], df_aggregated['distance'], 3)
    poly_func_dist_model = np.poly1d(coeffs_dist_model)
    dc0, dc1, dc2, dc3 = coeffs_dist_model
    print(f"Equation: distance ≈ {dc0:.8f}*pitch³ + {dc1:.8f}*pitch² + {dc2:.8f}*pitch + {dc3:.8f}") # 방정식 표현 (영어)

    plt.figure(figsize=(12, 7))
    plt.scatter(df_aggregated['pitch'], df_aggregated['distance'], color='blue', label='Mean Distance per Pitch (Data)')
    pitch_range_for_plot_dist = np.linspace(df_aggregated['pitch'].min(), df_aggregated['pitch'].max(), 200)
    distance_predicted_values = poly_func_dist_model(pitch_range_for_plot_dist)
    plt.plot(pitch_range_for_plot_dist, distance_predicted_values, color='red', 
             label=f'Polynomial Fit (degree 3, distance = f(pitch))')
    plt.title('Mean Projectile Distance vs. Pitch (distance = f(pitch) Model)') # 그래프 제목 (영어)
    plt.xlabel('Pitch (degrees)')                   # X축 레이블 (영어)
    plt.ylabel('Mean Distance')                     # Y축 레이블 (영어)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 방정식을 사용하여 특정 거리에 대한 pitch 계산 예시
    print("\n--- Example of Using the Equation ---") # 프로그램 출력은 영어로 유지
    example_distance = 75.0
    calculated_pitch = poly_func_pitch_model(example_distance)
    print(f"For a target distance of {example_distance}, calculated pitch using the equation: {calculated_pitch:.2f} degrees") # 프로그램 출력은 영어로 유지


if __name__ == '__main__':
    # WSL 환경에서의 파일 경로
    # 사용자의 Windows 경로: C:\Users\bok7z\0602_bullet.csv
    # WSL 경로로 변환: /mnt/c/Users/bok7z/0602_bullet.csv
    file_path_wsl = "/mnt/c/Users/bok7z/0602_bullet.csv"
    analyze_bullet_data(file_path_wsl)
