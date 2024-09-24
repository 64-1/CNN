folder_path = os.path.join(os.getcwd(), 'test')

a_values = []
b_values = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    img = Image.open(file_path)
    rgb = np.array(img).astype(float)/255 #normalise RGB to [0, 1]
    print('rgb.shape:', rgb.shape)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    if rgb.ndim == 2:
        rgb = np.stack((rgb)*3, axis=-1)

    pixels = rgb.reshape(-1, 3) #Reashape the image array 

    print('pixel.shape:', pixels.shape)

    #apply gamma correction
    """
    Gamma Correction (sRGB to Linear RGB):
    sRGB values are converted to linear RGB using the standard gamma correction formula.
    This steps corrects for the non-linear perception of brightness by human eye.
    """
    def gamma_correction(channel):
        mask = channel <= 0.04045
        channel[mask] = channel[mask] /12.92
        channel[~mask] = ((channel[~mask] + 0.055)/1.055) ** 2.4
        return channel

    linear_rgb = gamma_correction(pixels.copy())
    
    # Convert linear rgb to xyz
    """
    Linear RGB values are transformed into the XYZ color space using a matrix transformation.
    The XYZ color space is device-independent, which serves as an intermediary in color conversion
    """
    X = 0.4124564 * linear_rgb[:, 0]
    Y = 0.2126729 * linear_rgb[:, 0]
    Z = 0.0193339 * linear_rgb[:, 0]

    # Reference white point
    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.08883

    """
    XYZ values are normalised by the reference white point (D65 standard illuminant)
    to adjust for different lighting conditions
    """
    # Normalize XYZ
    x = X/Xn
    y = Y/Yn
    z = Z/Zn

    """
    The function f(t) accounts for the non-linear relationship in the LAB color space.
    L*, a* and b* are calculated using the normalised XYZ values
    """
    def f(t):
        delta = 6/29
        return np.where(t > delta ** 3, t** (1/3), (t/(3 * delta ** 2)) + (4/29))
    
    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = (116 * fy) -16
    a = 500 * (fx-fy)
    b = 200 * (fy -fz)

    L = np.clip(L, 0, 100)
    a = np.clip(a, -128, 127)
    b = np.clip(b, -128, 127)

    num_pixels = len(a)
    print(f'Total number of pixels: {num_pixels}')

    lab_data = np.column_stack((a, b))

    k=3

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(lab_data)

    labels = kmeans.labels_

    # Print cluster centers
centers = kmeans.cluster_centers_
for idx, center in enumerate(centers):
    print(f"Cluster {idx} center: a* = {center[0]:.2f}, b* = {center[1]:.2f}")

    # Reshape labels to match the image dimensions
labels_image = labels.reshape(rgb.shape[0], rgb.shape[1])

# Define colors for each cluster (you can customize these colors)
cluster_colors = np.array([
    [255, 255, 255],  # White for background (Cluster 0)
    [255, 0, 0],      # Red for stained region (Cluster 1)
    [184, 115, 51],   # Brownish for copper color (Cluster 2)
], dtype=np.uint8)

# Create an empty image to hold the segmented image
segmented_image = cluster_colors[labels_image]

# Display the segmented image
plt.figure(figsize=(8, 8))
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')
plt.show()

# Identify the stained cluster index (e.g., cluster with highest a* value)
stained_cluster_index = np.argmax(centers[:, 0])  # Index of cluster with highest a*

print(f"The stained cluster is Cluster {stained_cluster_index}")

# Create a mask for the stained region
stained_mask = (labels == stained_cluster_index)

# Extract a* values for the stained region
stained_a_values = a[stained_mask]

# Compute the average a* value
average_a_value = np.mean(stained_a_values)
print(f"The average a* value for the stained region is: {average_a_value:.2f}")

    df = pd.DataFrame({'a*': a, 'b*': b, 'Cluster': labels})

    # value_counts counts the number of occurences of each cluster label.
    cluster_counts = df['Cluster'].value_counts().sort_index()

    # The counts are printed
    for cluster_num, count in cluster_counts.items():
        print(f'Cluster {cluster_num}: {count} pixels')

    # Get k number of distinct colors for the color map
    colors = plt.get_cmap('tab10', k)

    plt.figure(figsize=(16,16))
    scatter = plt.scatter(df['a*'], df['b*'], c=labels, cmap=colors, s=1)
    plt.xlabel('a*')
    plt.ylabel('b*')
    plt.title('trial')
    plt.xlim(-128, 127)
    plt.ylim(-128, 127)
    plt.grid(True)
    plt.colorbar(scatter, ticks=range(k), label='Cluster')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s= 100, alpha = 0.6, marker='X')

    for i, (center_x, center_y) in enumerate(centers):
        count = cluster_counts[i]
        plt.text(center_x, center_y, f'{count}', color='red', fontsize=9,
                 ha='center', va='center', weight='bold')
        
    plt.show()
