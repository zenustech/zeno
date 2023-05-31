import pandas as pd
import numpy as np
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_keyword(df, keyword, n=3, engine='text-embedding-ada-002'):
    print(f'==> Searching for {keyword}...')
    key_embed = get_embedding(keyword, engine=engine)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, key_embed))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined
    )
    for i, r in enumerate(results):
        print(f'==> Found chunk {i + 1}:')
        print(r[:100], '...')
        print()
    return results

def chunk_split(article, chunk_size):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(article)
    chunks = []

    current_chunk = []
    current_token_count = 0

    for token in tokens:
        if current_token_count + 1 <= chunk_size:
            current_chunk.append(token)
            current_token_count += 1
        else:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = [token]
            current_token_count = 1

    if current_chunk:
        chunks.append(encoding.decode(current_chunk))

    return chunks

def embed_article(article, chunk_size=400, engine='text-embedding-ada-002'):
    # sections = []
    # for chunk_start in range(0, len(article), chunk_size):
    #     start = max(0, chunk_start - chunk_overlap)
    #     end = min(len(article), chunk_start + chunk_size + chunk_overlap)
    #     chunk = article[start:end]
    #     # print(f'==> Chunk {start} to {end}')
    #     sections.append(chunk)
    sections = chunk_split(article, chunk_size)

    df = pd.DataFrame(np.array(sections), columns=['combined'])
    counter = 0
    def do_embed(x):
        nonlocal counter
        counter += 1
        print(f'==> Embedding chunk {counter}...')
        return get_embedding(x, engine=engine)
    df["embedding"] = df.combined.apply(do_embed)
    return df

if __name__ == '__main__':
    article = '''Figure 3 shows an example in which the sphere is reflected a couple of times by the two mirrors before its image finally reaches the eye. What about transparent objects? We should see through transparent surfaces, though materials such as water or glass bend light rays due to the phenomenon of refraction. The optical (and physical) laws governing the phenomena of reflection and refraction are well known. The reflection direction only depends on the surface orientation and the incoming light direction. The refraction direction can be computed using Snell's law and depends on the surface orientation (the surface's normal), the incoming light direction, and the material refractive index (around 1.3 for water and 1.5 for glass). If you know these parameters, computing these directions is just a straight application of the laws of reflection and refractions whose equations are quite simple.
    Whitted proposed to use these laws to compute the reflection and refraction direction of rays as they intersect reflective or transparent surfaces, and follow the path of these rays to find out the color of the object they would intersect. Three cases may occur at the intersection point of these reflection and refraction rays (which in CG, we call secondary rays to differentiate them from the primary or camera rays):
        Case 1: if the surface at the intersection point is opaque and diffuse, all we need to do is use an illumination model such as the Phong model, to compute the color of the object at the intersection point. This process also involves casting a ray in the direction of each light in the scene to find if the point is in shadow. These rays are called shadow rays.
        Case 2: If the surface is a mirror-like surface, we simply trace another reflection ray at the intersection point.
        Case 3: if the surface is transparent, we then cast another reflection and a refraction ray at the intersection point.
    If P1 generates a secondary ray that intersects P2 and another secondary ray is generated at P2 that intersects another object at P3, then the color at P3 becomes the color of P2 which in turn becomes the color of P1 (assuming P1 and P2 belong to a mirror-like surface) as shown in Figure 6. The higher the refractive index of a transparent object is, the stronger these specular reflections are. Additionally, the higher the angle of incidence is, the more light is reflected (this is due to the Fresnel effect). The same method is used for refraction rays. If the ray is refracted a P1 then refracted at P2 and then finally hits a diffuse object at P3, P2 takes on the color of P3 and P1 takes on the color of P2 as shown in Figure 7. The higher the refractive index, the stronger the distortion (figure 3).
        Note that Whitted's algorithm uses backward tracing to simulate reflection and refraction, which as mentioned in the previous chapter, is more efficient than forward tracing. We start from the eye, cast primary rays, find the first visible surface and either compute the color of that surface at the intersection point if the object is opaque and diffuse or simply cast a reflection or a reflection and a refraction ray if the surface is either reflective (like a mirror) or transparent.
        Note also that Whitted's algorithm is a light transport algorithm. It can be used to simulate the appearance of diffuse objects (if you use the Phong illumination you would get a mix of diffuse and specular reflection but this is off-topic for now), mirror-like and transparent objects (glass, water, etc.). We will learn how to categorize light transport algorithms based on the sort of lighting effects they can simulate in the introductory lesson on shading [link] (this is very important to see the differences between each algorithm). For example, the Whitted algorithm simulates diffuse and glossy surfaces as well as specular reflection and refraction.
        Finally, and that is probably the most important part, note that Whitted's algorithm relies on computing visibility between points to simulate effects such as reflection, refraction, or shadowing. Once the intersection point P1 is found, we can compute the reflection or refraction direction using the law of reflection or refraction. We then get a new ray direction (or two new directions if the surface is transparent). The next step requires casting this ray into the scene and finding the next intersection point if any (P2 in our examples). And naturally, Whitted used ray-tracing in his particular implementation to compute the visibility between points (between the eye and P1, P1 and P2, P2 and P3, and finally P3 and the light in our examples). On a general note, Whitted was the first to use ray tracing to simulate these effects. Not surprisingly, his technique got a lot of success because it allowed the simulation of physically accurate reflections and refractions for the first time (figure 2).
    The paper was published in 1980, but it took about twenty more years before ray tracing started to get used for anything else than just research projects, due to its high computational cost. It is only in the late 1990s and early 2000s that ray tracing started to be used in production (for films).
    Some program uses a hybrid approach. They use rasterization and z-buffering to compute visibility with the first visible surface from the eye, and ray-tracing to simulate effects such as reflection and refraction.
    The Whitted algorithm is the classical example of an algorithm that uses ray tracing to produce photo-realistic computer-generated images. Many more advanced light transport algorithms have been developed since the paper was first published. Let's now review some of the algorithm's properties. Each time a ray intersects a transparent surface, two new rays are generated (a reflection and a refraction ray). If these rays intersect with another transparent object, each ray will generate two more rays. This process can go on forever, as long as secondary rays keep intersecting reflective or transparent objects. In this particular scenario, the number of rays increases exponentially as the recursion depth increases. The primary ray generates 2 secondary rays. This is the first level of recursion (depth 1). After two levels of recursion (depth 2), 4 rays are generated. At depth 3, each ray produces 2 more rays, thus producing a total of 8 rays. At depth 4, we have 16 rays and at depth 5, 32 rays. This practically means that render time also increases exponentially as the depth increases. The simplest solution to the recursive nature of this algorithm and the fact that the number of secondary rays grows exponentially as the depth increases is to put a cap on the maximum number of allowed recursions (or maximum depth). As soon as the maximum depth is reached, then we simply stop generating new rays.
    '''
    df = embed_article(article)
    results = search_keyword(df, "Whitted Ray-Tracing Algorithm", n=1)
