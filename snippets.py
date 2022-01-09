# Snippets

# Save predictions matrix
plt.matshow(similarity.float().cpu().numpy())
plt.savefig('matrix.png')